from tests import verify_utils


def test_deep_verify_backend_flag() -> None:
    args = verify_utils.DeepVerifyArgs(
        backend="pycmg",
        vg_start=0.0,
        vg_stop=0.0,
        vg_step=1.0,
        vd_start=0.0,
        vd_stop=0.0,
        vd_step=1.0,
    )
    assert verify_utils.run_deep_verify(args)
