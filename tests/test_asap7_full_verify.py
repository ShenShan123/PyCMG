from __future__ import annotations

from tests import verify_utils


def test_asap7_full_verify() -> None:
    args = verify_utils.DeepVerifyArgs(
        backend=verify_utils.BACKEND_PYCMG,
        tran=True,
        vg_start=0.0,
        vg_stop=1.2,
        vg_step=0.1,
        vd_start=0.0,
        vd_stop=1.2,
        vd_step=0.1,
    )
    assert verify_utils.run_asap7_full_verify(args)
