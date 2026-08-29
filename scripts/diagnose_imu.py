"""IMU diagnostic battery — checks the BNO085 for the systematic biases that
domain randomization cannot absorb and simulation cannot see.

Why this exists: the training sim has a perfect, perfectly-aligned IMU, so a
systematic bias or a wrong axis remap on the real robot is invisible to every
sim metric while still destabilizing the real robot (a constant accel-X error
reads as a permanent phantom pitch lean: ~5.8 deg per 1 m/s^2). Training-time
IMU randomization is zero-centered, so it trains tolerance to noise magnitude,
not to a constant offset.

Extra risk specific to this robot: the IMU was ported BNO055 -> BNO085
(commit f65c95f). The software axis remap in raw_imu.py faithfully reproduces
the BNO055's old hardware axis_remap, but those constants were derived for the
BNO055's chip-frame convention and mounting. Whether they are correct for a
BNO085 in this robot has never been verified physically. Test 2 settles that.

Three tests, in order of how few assumptions they make:

  1. INVARIANTS   - pose-independent. |accel| must be ~9.81 in any stationary
                    orientation, and gyro must be ~0. Valid even hand-held,
                    lying down, whatever, so long as it is momentarily still.
  2. AXES         - 6-position test. Determines which body axis reads gravity
                    in each orientation, which fully validates the remap
                    without needing any reference pose. Hand-held is fine.
  3. STANDING     - the only pose-specific test, and the only one that can
                    measure the phantom-lean bias directly. Requires the robot
                    free-standing in its home pose, untouched, on level ground.

Because the BNO085 self-calibrates continuously (unlike the BNO055, whose
calibration was stored and restored explicitly), its bias can differ between
sessions. A single clean run does NOT clear the IMU — use --log and repeat
across several power cycles.

Usage:
    uv run scripts/diagnose_imu.py                    # all tests
    uv run scripts/diagnose_imu.py --test invariants  # just the quick ones
    uv run scripts/diagnose_imu.py --test standing --log imu_runs.jsonl

This script only READS the IMU; it never commands a servo. For the standing
test, put the robot in its home pose first (scripts/turn_on.py) and let go.
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.imu_axis_remap import resolve as resolve_remap
from mini_bdx_runtime.raw_imu import SUPPORTED_CHIPS, make_imu

GRAVITY = 9.81

# Expected accelerometer reading with the robot free-standing in the home pose,
# measured in simulation (scene_flat_terrain_backlash.xml, home keyframe,
# settled 2000 steps). The non-zero X is REAL: it is the gravity component from
# the trunk's home-pose pitch, not a sensor bias. This is exactly why
# raw_imu.tare_x() must not be used in this pose — it would zero out a
# legitimate signal and manufacture a ~2.2 deg lean the other way.
EXPECTED_STANDING_ACCEL = np.array([-0.372, -0.004, 9.803])

# Thresholds
STILL_STD_MAX = 0.15  # m/s^2; above this the robot was not actually still
GRAV_TOL_PASS = 0.30  # |accel| deviation from 9.81
GRAV_TOL_WARN = 0.60
GYRO_PASS = 0.02  # rad/s at rest
GYRO_WARN = 0.05
ACCEL_BIAS_PASS = 0.20  # m/s^2 deviation from expected standing vector
ACCEL_BIAS_WARN = 0.50

# 6-position test: (prompt, expected dominant axis index, expected sign).
# Body frame is x=forward, y=left, z=up. A MuJoCo-style accelerometer at rest
# reads +g along whichever body axis points UP.
POSITIONS = [
    ("UPRIGHT — normal standing orientation", 2, +1),
    ("INVERTED — upside down", 2, -1),
    ("NOSE DOWN — front/beak pointing at the floor", 0, -1),
    ("NOSE UP — front/beak pointing at the ceiling", 0, +1),
    ("LEFT SIDE DOWN — robot's own left toward floor", 1, -1),
    ("LEFT SIDE UP — robot's own left toward ceiling", 1, +1),
]
AXIS_NAMES = ["X", "Y", "Z"]


def verdict(value, pass_thr, warn_thr):
    if value <= pass_thr:
        return "PASS"
    if value <= warn_thr:
        return "WARN"
    return "FAIL"


def sample(imu, n, settle):
    """Collect n fresh readings; returns (accel_mean, accel_std, gyro_mean)."""
    if settle:
        time.sleep(settle)
    acc, gyr = [], []
    while len(acc) < n:
        d = imu.get_data()
        acc.append(np.asarray(d["accelero"], dtype=float))
        gyr.append(np.asarray(d["gyro"], dtype=float))
        time.sleep(0.03)  # IMU thread runs at sampling_freq (50 Hz)
    acc = np.array(acc)
    gyr = np.array(gyr)
    return acc.mean(axis=0), acc.std(axis=0).max(), gyr.mean(axis=0)


def fmt(v):
    return f"[{v[0]:+7.3f} {v[1]:+7.3f} {v[2]:+7.3f}]"


def check_still(std):
    if std > STILL_STD_MAX:
        print(
            f"  ! NOT STILL (accel std {std:.3f} > {STILL_STD_MAX}) — "
            "readings unreliable, hold steadier and retry"
        )
        return False
    return True


def test_invariants(imu, args):
    """Pose-independent: |accel| must be ~g and gyro ~0 in ANY still pose."""
    print("\n=== TEST 1: INVARIANTS (pose-independent) ===")
    print(
        "Hold the robot still in a few DIFFERENT arbitrary orientations.\n"
        "Hand-held is fine — it only needs to be momentarily still.\n"
        "These checks do not depend on pose, so they are the most trustworthy."
    )
    results = []
    for i in range(args.poses):
        input(f"\n  Pose {i + 1}/{args.poses}: hold still, then press Enter...")
        a, std, g = sample(imu, args.samples, args.settle)
        still = check_still(std)
        mag = float(np.linalg.norm(a))
        gmag = float(np.linalg.norm(g))
        v_g = verdict(abs(mag - GRAVITY), GRAV_TOL_PASS, GRAV_TOL_WARN)
        v_gy = verdict(gmag, GYRO_PASS, GYRO_WARN)
        print(f"    accel {fmt(a)}  |accel| = {mag:6.3f}  [{v_g}]")
        print(f"    gyro  {fmt(g)}  |gyro|  = {gmag:6.4f}  [{v_gy}]")
        results.append(
            {"accel": a.tolist(), "gyro": g.tolist(), "accel_mag": mag,
             "gyro_mag": gmag, "still": still, "grav": v_g, "gyro_v": v_gy}
        )

    mags = [r["accel_mag"] for r in results]
    worst_g = max(abs(m - GRAVITY) for m in mags)
    worst_gy = max(r["gyro_mag"] for r in results)
    print("\n  --- summary ---")
    print(f"  worst |accel| error : {worst_g:.3f} m/s^2  [{verdict(worst_g, GRAV_TOL_PASS, GRAV_TOL_WARN)}]")
    print(f"  worst |gyro| at rest: {worst_gy:.4f} rad/s  [{verdict(worst_gy, GYRO_PASS, GYRO_WARN)}]")
    if worst_g > GRAV_TOL_WARN:
        print("  -> |accel| is off in some pose: scale/units/calibration problem,")
        print("     NOT just a mounting bias. Fix this before anything else.")
    if worst_gy > GYRO_WARN:
        print("  -> gyro reads non-zero at rest: gyro bias. The policy consumes")
        print("     gyro directly, so a constant offset is a constant false rotation.")
    return {"poses": results, "worst_accel_err": worst_g, "worst_gyro": worst_gy}


def test_axes(imu, args):
    """6-position test: does each body axis read gravity where it should?"""
    print("\n=== TEST 2: AXIS / REMAP (6-position) ===")
    print(
        "This validates the BNO055-inherited axis remap against the BNO085.\n"
        "Hold the robot in each orientation — rough is fine, it just needs to be\n"
        "closer to the named axis than to any other, and steady."
    )
    results = []
    n_fail = 0
    for prompt, exp_axis, exp_sign in POSITIONS:
        input(f"\n  {prompt}\n    press Enter when steady...")
        a, std, _ = sample(imu, args.samples, args.settle)
        check_still(std)
        dom = int(np.argmax(np.abs(a)))
        sign = 1 if a[dom] >= 0 else -1
        ok = (dom == exp_axis) and (sign == exp_sign)
        n_fail += 0 if ok else 1
        exp_s = f"{'+' if exp_sign > 0 else '-'}{AXIS_NAMES[exp_axis]}"
        got_s = f"{'+' if sign > 0 else '-'}{AXIS_NAMES[dom]}"
        print(f"    accel {fmt(a)}   expected {exp_s:>3}  got {got_s:>3}  "
              f"[{'PASS' if ok else 'FAIL'}]")
        results.append({"pose": prompt, "accel": a.tolist(),
                        "expected": exp_s, "got": got_s, "ok": ok})

    print("\n  --- summary ---")
    if n_fail == 0:
        print("  All 6 positions correct — the remap is valid for this BNO085.")
    else:
        print(f"  {n_fail}/6 positions WRONG — the inherited BNO055 remap does not")
        print("  match this chip/mounting. Fix raw_imu._remap_vector() (or the")
        print("  imu_upside_down flag) before trusting ANY policy on hardware.")
        print("  Map each 'got' back to the intended axis to derive the correct remap.")
    return {"positions": results, "n_fail": n_fail}


def test_standing(imu, args):
    """Pose-specific: measure the phantom-lean bias against the sim reference."""
    print("\n=== TEST 3: STANDING BIAS (pose-specific) ===")
    print(
        "Requires ALL of the following, or the number is meaningless:\n"
        "  - servos powered and holding the home pose (scripts/turn_on.py)\n"
        "  - robot free-standing on its own feet, NOT held\n"
        "  - ground verified level (a 2 deg slope fakes exactly this error)\n"
        "  - robot completely stationary"
    )
    input("\n  Press Enter when the robot is standing untouched...")
    a, std, g = sample(imu, args.samples, args.settle)
    still = check_still(std)

    diff = a - EXPECTED_STANDING_ACCEL
    print(f"\n    expected {fmt(EXPECTED_STANDING_ACCEL)}   (from sim home pose)")
    print(f"    measured {fmt(a)}")
    print(f"    diff     {fmt(diff)}")

    print("\n  --- per-axis ---")
    lean = {}
    for i, name in enumerate(AXIS_NAMES):
        d = abs(diff[i])
        v = verdict(d, ACCEL_BIAS_PASS, ACCEL_BIAS_WARN)
        # X error -> phantom pitch, Y error -> phantom roll
        note = ""
        if i in (0, 1):
            deg = float(np.degrees(np.arctan2(abs(diff[i]), GRAVITY)))
            lean["pitch" if i == 0 else "roll"] = deg
            note = f"  ~= {deg:.2f} deg phantom {'pitch' if i == 0 else 'roll'}"
        print(f"    {name}: diff {diff[i]:+7.3f}  [{v}]{note}")

    worst = float(np.abs(diff).max())
    overall = verdict(worst, ACCEL_BIAS_PASS, ACCEL_BIAS_WARN)
    print(f"\n  --- summary --- [{overall}]")
    if overall == "PASS":
        print("  No significant standing bias. The IMU is probably not the cause")
        print("  of the fall/rock behaviour; look elsewhere.")
    else:
        pitch_deg = lean.get("pitch", 0.0)
        signed_pitch = float(np.degrees(np.arctan2(diff[0], GRAVITY)))
        print(f"  Systematic bias present. Phantom pitch lean ~{pitch_deg:.2f} deg.")
        print("  The policy will fight a tilt that does not exist — the known route")
        print("  to leaning, falling in the direction of travel, and growing rock.")
        print(f"  Suggested runtime correction:  --pitch_bias {signed_pitch:+.2f}")
        print("  (sign may need flipping — verify the lean shrinks, do not assume)")
        print("  Do NOT use raw_imu.tare_x(): it forces accel X to zero, deleting")
        print("  the legitimate -0.372 gravity component of the home pose.")
    return {"expected": EXPECTED_STANDING_ACCEL.tolist(), "measured": a.tolist(),
            "diff": diff.tolist(), "gyro": g.tolist(), "still": still,
            "worst": worst, "verdict": overall, "lean_deg": lean}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--test", choices=["all", "invariants", "axes", "standing"],
                   default="all")
    p.add_argument("--samples", type=int, default=100, help="readings averaged per measurement")
    p.add_argument("--settle", type=float, default=1.0, help="seconds to wait before sampling")
    p.add_argument("--poses", type=int, default=3, help="arbitrary poses for the invariant test")
    p.add_argument("--freq", type=int, default=50, help="IMU sampling frequency")
    p.add_argument("--log", type=str, default=None,
                   help="append results as JSON lines (use across power cycles "
                        "to check BNO085 self-calibration stability)")
    p.add_argument("--upside-down", dest="upside_down", action="store_true", default=None)
    p.add_argument("--no-upside-down", dest="upside_down", action="store_false")
    p.add_argument("--chip", choices=SUPPORTED_CHIPS, default=None,
                   help="override duck_config.json's imu_chip")
    p.add_argument("--axis-remap", default=None,
                   help="override the remap, e.g. 'y,x,-z' or a preset name. "
                        "Handy for testing a candidate mapping before editing config.")
    args = p.parse_args()

    cfg = DuckConfig(ignore_default=True)
    upside_down = cfg.imu_upside_down if args.upside_down is None else args.upside_down
    chip = args.chip or cfg.imu_chip
    remap_spec = args.axis_remap or cfg.imu_axis_remap
    remap = resolve_remap(chip, upside_down, remap_spec)

    src = lambda cli: "CLI" if cli is not None else "duck_config.json"
    print("IMU diagnostic")
    print(f"  chip        : {chip}  (from {src(args.chip)})")
    print(f"  upside_down : {upside_down}  (from {src(args.upside_down)})")
    print(f"  axis remap  : {remap.spec}  (from "
          f"{src(args.axis_remap) if remap_spec else 'chip+mounting preset'})")
    print("NOTE: the BNO085 self-calibrates continuously, so its bias can differ")
    print("between sessions. Repeat this across power cycles before concluding.")

    imu = make_imu(args.freq, upside_down=upside_down, chip=chip, axis_remap=remap)
    time.sleep(1.0)  # let the worker thread produce data

    out = {"timestamp": datetime.now().isoformat(), "chip": chip,
           "upside_down": upside_down, "axis_remap": remap.spec}
    try:
        if args.test in ("all", "invariants"):
            out["invariants"] = test_invariants(imu, args)
        if args.test in ("all", "axes"):
            out["axes"] = test_axes(imu, args)
        if args.test in ("all", "standing"):
            out["standing"] = test_standing(imu, args)
    except KeyboardInterrupt:
        print("\ninterrupted")

    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(out) + "\n")
        print(f"\nappended to {args.log}")


if __name__ == "__main__":
    main()
