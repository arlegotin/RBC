import json
from dataclasses import asdict, replace

import numpy as np

from rbc.data import ManifestView, SplitManifest, assign_splits, audit_dataset, build_policy_pools, renderer_for_split, returns_schema
from rbc.types import PrivateCase, PublicCase


def _write_case_files(tmp_path, names, text="No item condition was established."):
    schema = returns_schema()
    public_path = tmp_path / "public.jsonl"
    private_path = tmp_path / "private.jsonl"
    with public_path.open("w") as public_file, private_path.open("w") as private_file:
        for name, split in names:
            case = PublicCase(name, "returns-six-v1", schema, {}, text)
            private = PrivateCase(name, (False,) * 6, {"plan": {}, "results": []}, tuple([1 / 64] * 64), name, split, "plain", 1, ("missing",))
            public_file.write(json.dumps({"case_id": case.case_id, "schema_id": case.schema_id, "schema": [asdict(v) for v in case.schema], "records": {}, "text": case.text}) + "\n")
            private_file.write(json.dumps(asdict(private)) + "\n")
    return public_path, private_path


def test_assign_splits_is_deterministic_and_pilot_subset_stays_out_of_final():
    ids = [f"root-{i}" for i in range(10)]
    config = {"data": {"split_seed": 2718, "train": 4, "dev": 2, "calibration": 2, "test": 2}}
    manifest = assign_splits(ids, config)
    assert manifest.group_splits == assign_splits(ids, config).group_splits
    assert {split: list(manifest.group_splits.values()).count(split) for split in ("train", "dev", "calibration", "test")} == {"train": 4, "dev": 2, "calibration": 2, "test": 2}
    train = [id for id, split in manifest.group_splits.items() if split == "train"]
    dev = [id for id, split in manifest.group_splits.items() if split == "dev"]
    pilot = set(train[:2] + dev[:1])
    final = {id for id, split in manifest.group_splits.items() if split in {"calibration", "test"}}
    assert pilot.isdisjoint(final)


def test_reserved_calibration_and_test_share_frozen_renderer_mix():
    config = {"data": {"train_renderer_weights": {"plain": 1.0, "alternative": 0.0}, "dev_renderer_weights": {"plain": 0.25, "alternative": 0.75}, "target_renderer_weights": {"plain": 0.25, "alternative": 0.75}}}
    assert all(renderer_for_split(config, "train", np.random.default_rng(seed)) == "plain" for seed in range(10))
    for seed in range(20):
        assert renderer_for_split(config, "calibration", np.random.default_rng(seed)) == renderer_for_split(config, "test", np.random.default_rng(seed))


def test_independent_recurrence_is_recorded_but_not_rejected(tmp_path):
    names = [("case-a", "calibration"), ("case-b", "test")]
    public_path, private_path = _write_case_files(tmp_path, names)
    manifest = SplitManifest({name: split for name, split in names}, views=(ManifestView("case-a", "case-a", "calibration", None, "plain", True), ManifestView("case-b", "case-b", "test", None, "plain", True)))
    audit = audit_dataset(public_path, private_path, manifest, build_policy_pools(returns_schema(), 1))
    assert audit.ok, audit.errors
    assert any("recurrence" in note for note in audit.notes)


def test_copied_child_cross_split_and_duplicate_primary_are_rejected(tmp_path):
    names = [("case-a", "calibration"), ("case-b", "test")]
    public_path, private_path = _write_case_files(tmp_path, names)
    manifest = SplitManifest({name: split for name, split in names}, views=(ManifestView("case-a", "case-a", "calibration", None, "plain", True), ManifestView("case-b", "case-b", "test", "case-a", "plain", True)))
    audit = audit_dataset(public_path, private_path, manifest, build_policy_pools(returns_schema(), 1))
    assert not audit.ok
    assert any("parent" in error or "copied" in error for error in audit.errors)
    manifest = replace(manifest, views=(ManifestView("case-a", "case-a", "calibration", None, "plain", True), ManifestView("case-b", "case-a", "calibration", "case-a", "plain", True)))
    audit = audit_dataset(public_path, private_path, manifest, build_policy_pools(returns_schema(), 1))
    assert not audit.ok
    assert any("primary" in error or "group" in error for error in audit.errors)
