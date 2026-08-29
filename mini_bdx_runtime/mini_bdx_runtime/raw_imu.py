"""IMU factory — selects the backend from configuration.

The concrete implementations live in bno085_imu.py and bno055_imu.py; the axis
mapping lives in imu_axis_remap.py. This module only picks one, and keeps the
historical `from mini_bdx_runtime.raw_imu import Imu` import path working.

Configure in duck_config.json:

    "imu_chip":        "bno085" | "bno055"     (default: bno085)
    "imu_upside_down": true | false            (default: false)
    "imu_axis_remap":  "y,x,-z"                (optional; overrides the
                                                chip+mounting preset, and also
                                                accepts a preset name)

The chip modules are imported lazily so that selecting one chip does not
require the other's driver to be installed.
"""

from mini_bdx_runtime.imu_axis_remap import PRESETS, AxisRemap, resolve  # noqa: F401

SUPPORTED_CHIPS = ("bno085", "bno055")


def make_imu(
    sampling_freq,
    user_pitch_bias=0,
    calibrate=False,
    upside_down=True,
    chip="bno085",
    axis_remap=None,
):
    """Build the configured IMU backend."""
    chip = (chip or "bno085").strip().lower()
    if chip not in SUPPORTED_CHIPS:
        raise ValueError(
            f"Unknown imu_chip '{chip}'. Supported: {list(SUPPORTED_CHIPS)}"
        )

    # Resolve (and validate) the remap here so a bad spec fails immediately with
    # a clear message, before any I2C work.
    remap = resolve(chip, upside_down, axis_remap)

    if chip == "bno085":
        from mini_bdx_runtime.bno085_imu import Bno085Imu as Backend
    else:
        from mini_bdx_runtime.bno055_imu import Bno055Imu as Backend

    return Backend(
        sampling_freq,
        user_pitch_bias=user_pitch_bias,
        calibrate=calibrate,
        upside_down=upside_down,
        axis_remap=remap,
    )


def from_config(duck_config, sampling_freq, user_pitch_bias=0, calibrate=False):
    """Build the IMU straight from a DuckConfig."""
    return make_imu(
        sampling_freq,
        user_pitch_bias=user_pitch_bias,
        calibrate=calibrate,
        upside_down=duck_config.imu_upside_down,
        chip=duck_config.imu_chip,
        axis_remap=duck_config.imu_axis_remap,
    )


# Backwards-compatible entry point: `Imu(...)` used to be the BNO085 class.
# Callers that predate the chip option keep working and get the default chip.
Imu = make_imu


if __name__ == "__main__":
    import argparse
    import time

    import numpy as np

    p = argparse.ArgumentParser(description="Print live IMU readings.")
    p.add_argument("--chip", default="bno085", choices=SUPPORTED_CHIPS)
    p.add_argument("--upside-down", dest="upside_down", action="store_true", default=False)
    p.add_argument("--axis-remap", default=None, help="e.g. 'y,x,-z' or a preset name")
    args = p.parse_args()

    imu = make_imu(
        50, upside_down=args.upside_down, chip=args.chip, axis_remap=args.axis_remap
    )
    while True:
        data = imu.get_data()
        print("gyro", np.around(data["gyro"], 3))
        print("accelero", np.around(data["accelero"], 3))
        print("---")
        time.sleep(1 / 25)
