"""Move the robot to a pose saved as JSON (e.g. by capture_pose.py).

The pose file is either the format written by capture_pose.py:

    {"name": "wave", "joints": {"left_hip_yaw": 0.0, ...}}

or a bare mapping of joint_name -> angle (radians):

    {"left_hip_yaw": 0.0, ...}

The robot first goes through the normal startup (init pose), then smoothly
interpolates from there to the target pose.

Examples:
    python play_pose.py poses/wave.json
    python play_pose.py poses/wave.json --duration 1.5
"""

import argparse
import json
import os
import sys
import time

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI

HOME_DIR = os.path.expanduser("~")


def load_pose(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Accept both {"joints": {...}} and a bare {joint: angle} mapping.
    joints = data.get("joints", data) if isinstance(data, dict) else None
    if not isinstance(joints, dict):
        raise ValueError(f"{path} does not contain a joint->angle mapping")
    return {name: float(angle) for name, angle in joints.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Move the robot to a pose saved as JSON."
    )
    parser.add_argument("pose_file", help="path to the JSON pose file")
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="serial port of the motor bus (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--duck_config_path",
        default=f"{HOME_DIR}/duck_config.json",
        help="path to duck_config.json (for joint offsets)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="seconds to interpolate from the init pose to the target (default: 2.0)",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=50.0,
        help="interpolation update rate in Hz (default: 50)",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="keep torque on and hold the pose until Ctrl-C "
        "(default: exit immediately, leaving torque on)",
    )
    args = parser.parse_args()

    target = load_pose(args.pose_file)

    duck_config = DuckConfig(config_json_path=args.duck_config_path)
    hwi = HWI(duck_config, args.port)

    # Iterate joints in the HWI's canonical order (dict preserves order).
    joint_order = list(hwi.joints.keys())
    motor_ids = list(hwi.joints.values())
    valid = set(joint_order)
    unknown = [j for j in target if j not in valid]
    if unknown:
        print(f"WARNING: ignoring unknown joints: {unknown}", file=sys.stderr)
        target = {j: a for j, a in target.items() if j in valid}

    # Read where the robot currently is (works even with torque disabled) so we
    # can interpolate from the actual pose instead of snapping to init first.
    # Read immediately before energizing to minimize droop while still limp.
    current = hwi.get_present_positions()
    if current is None or len(current) != len(joint_order):
        print("ERROR: could not read current motor positions", file=sys.stderr)
        sys.exit(1)
    start = {j: float(a) for j, a in zip(joint_order, current)}

    # Joints not specified in the pose stay where they currently are.
    missing = [j for j in joint_order if j not in target]
    if missing:
        print(
            f"WARNING: joints not in pose, holding current position: {missing}",
            file=sys.stderr,
        )
    full_target = {j: target.get(j, start[j]) for j in joint_order}

    # Energize directly at full stiffness with the goal set to the current pose,
    # so the robot firms up in place. (A low->high gain ramp would let the limp
    # joints droop under gravity and then snap back when stiffened.)
    hwi.set_position_all(start)
    hwi.io.set_kps(motor_ids, hwi.kps)

    # Smoothly interpolate from the current pose to the target pose.
    steps = max(1, int(args.duration * args.freq))
    print(f"Moving to pose '{os.path.basename(args.pose_file)}'...", file=sys.stderr)
    for i in range(1, steps + 1):
        alpha = i / steps
        blended = {
            j: (1 - alpha) * start[j] + alpha * full_target[j] for j in joint_order
        }
        hwi.set_position_all(blended)
        time.sleep(1.0 / args.freq)

    print("Pose reached.", file=sys.stderr)

    if args.hold:
        print("Holding pose (Ctrl-C to release)...", file=sys.stderr)
        try:
            while True:
                hwi.set_position_all(full_target)
                time.sleep(1.0 / args.freq)
        except KeyboardInterrupt:
            hwi.turn_off()
            print("\nTorque disabled.", file=sys.stderr)


if __name__ == "__main__":
    main()
