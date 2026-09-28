import json
from dataclasses import replace

import numpy as np

from conftest import schema_n, var
from rbc.logic import decide, load_bundle, make_bundle, save_bundle
from rbc.types import Accept, BundleProvenance, OutOfScope, SupportBelief


P = BundleProvenance("evidence", "records", "schema", "tokenizer", "encoder", "head", "calibration")


def test_saved_bundle_replays_and_new_policy_needs_no_compiler(tmp_path):
    schema = schema_n(2)
    support = np.array([False, True, True, False])
    save_bundle(make_bundle(schema, SupportBelief(support), support, P), tmp_path / "bundle")
    loaded = load_bundle(tmp_path / "bundle", P)
    assert isinstance(decide(loaded, {"op": "count_eq", "ids": ["v0", "v1"], "k": 1}), Accept)
    assert decide(loaded, var("v0")).world_a == 1


def test_changed_provenance_or_corrupted_array_never_reuses_bundle(tmp_path):
    schema = schema_n(2)
    support = np.array([False, True, True, False])
    path = tmp_path / "bundle"
    save_bundle(make_bundle(schema, SupportBelief(support), support, P), path)
    for key in ("evidence_hash", "records_hash", "schema_hash", "tokenizer_hash", "encoder_hash", "head_hash", "calibration_hash"):
        assert isinstance(load_bundle(path, replace(P, **{key: "changed"})), OutOfScope)
    meta = json.loads((path / "metadata.json").read_text())
    (path / meta["arrays_file"]).write_bytes(b"interrupted write")
    assert isinstance(load_bundle(path, P), OutOfScope)


def test_missing_metadata_is_incomplete_bundle(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()
    (path / "arrays-orphan.npz").write_bytes(b"orphan")
    assert isinstance(load_bundle(path, P), OutOfScope)
