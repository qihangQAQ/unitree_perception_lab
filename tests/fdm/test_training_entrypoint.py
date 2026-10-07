"""The online trainer must be safe for DataLoader's spawned Python processes."""

import subprocess
import sys
from pathlib import Path


def test_spawn_import_does_not_launch_simulator_or_parse_arguments():
    script = Path(__file__).resolve().parents[2] / "scripts" / "fdm" / "train_fdm.py"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "from pathlib import Path; "
            "script = sys.argv[1]; "
            "sys.path.insert(0, str(Path(script).parent)); "
            "runpy.run_path(script, run_name='__mp_main__'); "
            "assert 'isaaclab.app' not in sys.modules; "
            "assert 'unitree_rl_lab.tasks' not in sys.modules; "
            "assert 'torch' not in sys.modules",
            str(script),
            "--deliberately-invalid-worker-argument",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
