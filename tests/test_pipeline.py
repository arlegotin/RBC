import json
import subprocess
import sys


def test_later_gates_report_the_recorded_g1_stop():
    for gate in ("g2", "g3"):
        result = subprocess.run([sys.executable, "-m", "rbc", "run", "--through", gate, "--config", "configs/poc.yaml"], capture_output=True, text=True)
        assert result.returncode != 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "blocked_by_g1"
        assert payload["reason"] == "INSUFFICIENT_JOINT_MARGINAL_HEADROOM"
