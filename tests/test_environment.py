import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rbc.experiment import doctor, load_config


def test_cli_help_never_loads_optional_models():
    code = r'''
import builtins
import sys
old_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'mlx_lm'}:
        raise RuntimeError('eager optional import: ' + name)
    return old_import(name, *args, **kwargs)
builtins.__import__ = guarded
from rbc.__main__ import main
assert main(['--help']) == 0
assert not {'torch', 'transformers', 'mlx_lm'} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_config_rejects_unknown_keys(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("run_id: research-1\nunknown: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(config)


def test_doctor_reports_observed_hardware_without_optional_imports():
    result = doctor()
    assert result["architecture"] in {"arm64", "aarch64"}
    assert result["python"].startswith("3.")
    assert result["physical_memory_bytes"] > 0
    assert result["disk_free_bytes"] > 0
    assert result["physical_memory_source"]
