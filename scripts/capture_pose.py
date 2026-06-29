"""Read the current motor positions and output them as a pose in JSON.

Typical use is capturing a pose you set by hand: by default the motors are
made limp (torque disabled) so you can position the joints, then their angles
are read back and written out as JSON.

Examples:
    # Print the current pose to stdout
    python capture_pose.py

    # Pose the robot by hand, wait 3s, then save to a file
    python capture_pose.py --countdown 3 --name wave -o poses/wave.json

    # Read without disabling torque (capture whatever pose is being held)
    python capture_pose.py --keep-torque
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI

HOME_DIR = os.path.expanduser("~")


def read_pose(hwi, samples=10):
    """Read present positions a few times and average them for stability.

    Returns an ordered dict of joint_name -> angle (radians), or None if the
    motors could not be read.
    """
    joint_names = list(hwi.joints.keys())
    readings = []
    for _ in range(samples):
        positions = hwi.get_present_positions()
        if positions is None or len(positions) != len(joint_names):
            time.sleep(0.02)
            continue
        readings.append(positions)
        time.sleep(0.02)

    if not readings:
        return None

    mean = np.mean(readings, axis=0)
    return {name: round(float(angle), 4) for name, angle in zip(joint_names, mean)}


def main():
    parser = argparse.ArgumentParser(
        description="Read current motor positions and output them as a JSON pose."
    )
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
        "-o",
        "--output",
        default=None,
        help="file to write the JSON pose to (default: print to stdout)",
    )
    parser.add_argument(
        "--name",
        default="pose",
        help="name to label the pose with in the output (default: pose)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="number of reads to average for each joint (default: 10)",
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=0.0,
        help="seconds to wait before capturing, so you can pose the robot",
    )
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="do not disable torque before reading (default: motors go limp so "
        "you can pose them by hand)",
    )
    args = parser.parse_args()

    duck_config = DuckConfig(config_json_path=args.duck_config_path)
    hwi = HWI(duck_config, args.port)

    if not args.keep_torque:
        hwi.turn_off()  # disable torque so the joints can be moved by hand
        print("Torque disabled - pose the robot by hand.", file=sys.stderr)

    for remaining in range(int(args.countdown), 0, -1):
        print(f"Capturing in {remaining}...", file=sys.stderr)
        time.sleep(1)

    pose = read_pose(hwi, samples=args.samples)
    if pose is None:
        print("ERROR: could not read motor positions", file=sys.stderr)
        sys.exit(1)

    output = {"name": args.name, "joints": pose}
    text = json.dumps(output, indent=2)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Wrote pose '{args.name}' to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
