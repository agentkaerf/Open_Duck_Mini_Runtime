"""Configurable IMU axis remapping, shared by every IMU backend.

An IMU reports in its own chip frame. The trained policy's gyro/accel
observations assume the ROBOT BODY frame (x=forward, y=left, z=up). The mapping
between them depends on the chip's axis convention AND how the board is
physically mounted, so it belongs in configuration, not in code.

Spec format — three comma-separated terms giving the source of body x, y, z:

    "y,x,-z"   ->  body_x = +chip_y,  body_y = +chip_x,  body_z = -chip_z

Whitespace is ignored and '+' is optional. Any permutation of x/y/z with
independent signs is accepted, subject to one hard rule (see below).

WHY THE DETERMINANT IS VALIDATED
    Angular velocity (gyro) is an AXIAL vector; acceleration is a POLAR vector.
    Under a proper rotation (det=+1) both transform identically, so one remap
    can be applied to both — which is what every backend here does. Under a
    reflection (det=-1) they do NOT: the gyro would need an extra sign flip,
    and applying the same remap to both would silently invert angular rates
    while the accelerometer looked perfectly fine. That failure is close to
    undetectable by eye, so a det=-1 spec is rejected outright rather than
    warned about.

HISTORY (why this module exists)
    The BNO055 did this remap in hardware (its `axis_remap` register). When the
    IMU was ported to a BNO085 (commit f65c95f) those constants were carried
    over verbatim as a software remap — but they encoded the BNO055's chip
    frame and mounting, and were never re-derived for the BNO085. Measured on
    hardware 2026-08-28 (scripts/diagnose_imu.py, 6-position test): Z was
    correct but X and Y were both sign-inverted, i.e. the body frame was yawed
    180 deg. Every fore-aft and lateral balance correction was therefore applied
    with the wrong sign — positive feedback — which is invisible in simulation
    because the simulated IMU is perfectly aligned.
"""

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Named presets. Chip + mounting -> spec.
PRESETS = {
    # BNO085, board mounted upside down. VERIFIED on hardware 2026-08-28:
    # all six positions of the 6-position test pass with this spec.
    "bno085_upside_down": "y,x,-z",
    # BNO085, board mounted normally. Inherited from the BNO055 'normal' remap
    # and NOT yet verified on hardware — run the 6-position test before trusting.
    "bno085_normal": "-y,x,z",
    # BNO055 legacy specs, matching the original hardware axis_remap exactly
    # (AXIS_REMAP_Y, AXIS_REMAP_X, AXIS_REMAP_Z + sign flags).
    "bno055_upside_down": "-y,-x,-z",
    "bno055_normal": "-y,x,z",
    # No remap; chip frame == body frame.
    "identity": "x,y,z",
}


class AxisRemap:
    """A validated permutation-with-signs mapping chip axes to body axes."""

    def __init__(self, spec: str):
        self.spec = spec
        self.perm, self.signs = self._parse(spec)
        self.matrix = np.zeros((3, 3), dtype=float)
        for out_i, (src, sgn) in enumerate(zip(self.perm, self.signs)):
            self.matrix[out_i, src] = sgn

        det = float(np.linalg.det(self.matrix))
        if not np.isclose(det, 1.0):
            raise ValueError(
                f"IMU axis remap '{spec}' has determinant {det:+.0f}, but only "
                "proper rotations (det=+1) are allowed. A det=-1 spec is a "
                "reflection: it would transform the accelerometer correctly "
                "while silently INVERTING the gyro (angular velocity is an "
                "axial vector). Flip the sign of exactly one term to fix it."
            )

    @staticmethod
    def _parse(spec):
        terms = [t.strip().lower() for t in str(spec).split(",")]
        if len(terms) != 3:
            raise ValueError(
                f"IMU axis remap '{spec}' must have 3 comma-separated terms, "
                f"got {len(terms)}. Example: 'y,x,-z'"
            )
        perm, signs = [], []
        for t in terms:
            sign = 1
            if t.startswith("-"):
                sign, t = -1, t[1:]
            elif t.startswith("+"):
                t = t[1:]
            if t not in _AXIS_INDEX:
                raise ValueError(
                    f"IMU axis remap '{spec}': unknown axis '{t}'. "
                    "Each term must be x, y or z, optionally signed."
                )
            perm.append(_AXIS_INDEX[t])
            signs.append(sign)
        if sorted(perm) != [0, 1, 2]:
            raise ValueError(
                f"IMU axis remap '{spec}' must use each of x, y, z exactly once."
            )
        return perm, signs

    def apply(self, v):
        """Map a chip-frame 3-vector into the body frame."""
        v = np.asarray(v, dtype=float)
        return np.array([self.signs[i] * v[self.perm[i]] for i in range(3)])

    def to_bno055_axis_remap(self, module):
        """Translate to the BNO055's hardware axis_remap 6-tuple.

        The BNO055 applies the remap on-chip, so its driver takes
        (x_src, y_src, z_src, x_sign, y_sign, z_sign) rather than a matrix.
        `module` is the imported adafruit_bno055 (passed in so this file stays
        importable, and unit-testable, without the hardware libraries).
        """
        axis_const = (
            module.AXIS_REMAP_X,
            module.AXIS_REMAP_Y,
            module.AXIS_REMAP_Z,
        )
        pos, neg = module.AXIS_REMAP_POSITIVE, module.AXIS_REMAP_NEGATIVE
        return (
            axis_const[self.perm[0]],
            axis_const[self.perm[1]],
            axis_const[self.perm[2]],
            pos if self.signs[0] > 0 else neg,
            pos if self.signs[1] > 0 else neg,
            pos if self.signs[2] > 0 else neg,
        )

    def __repr__(self):
        return f"AxisRemap('{self.spec}')"


def resolve(chip: str, upside_down: bool, spec=None) -> AxisRemap:
    """Pick the remap: explicit spec wins, else the chip+mounting preset.

    `spec` may be a raw spec string ('y,x,-z') or a preset name.
    """
    if spec:
        return AxisRemap(PRESETS.get(str(spec).strip(), spec))
    key = f"{chip}_{'upside_down' if upside_down else 'normal'}"
    if key not in PRESETS:
        raise ValueError(
            f"No IMU axis-remap preset '{key}'. Known presets: "
            f"{sorted(PRESETS)}. Set 'imu_axis_remap' in duck_config.json "
            "to an explicit spec such as 'y,x,-z'."
        )
    return AxisRemap(PRESETS[key])
