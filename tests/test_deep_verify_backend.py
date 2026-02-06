from pathlib import Path
import subprocess
import sys


def test_deep_verify_backend_flag() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "deep_verify.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            "pycmg",
            "--vg-start",
            "0",
            "--vg-stop",
            "0",
            "--vg-step",
            "1",
            "--vd-start",
            "0",
            "--vd-stop",
            "0",
            "--vd-step",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
