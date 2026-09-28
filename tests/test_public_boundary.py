import json

import pytest

from conftest import schema_n
from rbc.types import PrivateCase, PublicCase, ValidationError, public_case_from_dict, public_model_input, read_public_jsonl


def test_private_world_requires_real_booleans():
    case = PrivateCase("case-1", (True, False), observation={}, oracle_posterior=(0.0, 1.0, 0.0, 0.0), source_group="root-1", split="train", renderer="plain", generation_seed=1, relation_tags=())
    assert case.true_index == 1
    with pytest.raises(ValidationError):
        PrivateCase("case-1", (1, False), observation={}, oracle_posterior=(0.0, 1.0, 0.0, 0.0), source_group="root-1", split="train", renderer="plain", generation_seed=1, relation_tags=())


def test_private_changes_do_not_change_model_input():
    public = PublicCase("case-1", "test-v1", schema_n(2), {}, "Exactly one item is damaged.")
    text = public_model_input(public)
    assert "case-1" not in text
    assert "Exactly one item is damaged." in text
    assert "true_world" not in text
    assert public_model_input(public) == public_model_input(public)


def test_public_reader_rejects_private_fields(tmp_path):
    valid = {"case_id": "case-1", "schema_id": "test-v1", "schema": [{"id": "v0", "entity": "item-0", "meaning": "item-0 is damaged"}], "records": {}, "text": "Damaged."}
    assert public_case_from_dict(valid).case_id == "case-1"
    with pytest.raises(ValidationError):
        public_case_from_dict({**valid, "true_world": [True]})
    with pytest.raises(ValidationError):
        public_case_from_dict({**valid, "records": {"private": {"latent_seed": 4}}})
    path = tmp_path / "public.jsonl"
    path.write_text(json.dumps(valid) + "\n")
    assert len(read_public_jsonl(path)) == 1
    path.write_text(json.dumps({**valid, "observation": {}}) + "\n")
    with pytest.raises(ValidationError):
        read_public_jsonl(path)
