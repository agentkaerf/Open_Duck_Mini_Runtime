import json
from typing import Optional
import os

HOME_DIR = os.path.expanduser("~")


class DuckConfig:

    @staticmethod
    def _feature_config(features, key):
        """Return a feature's config as a dict.

        Accepts the nested form ({"enable": ..., ...}) and, for backward
        compatibility, a plain bool (treated as {"enable": <bool>}).
        """
        value = features.get(key, {})
        if isinstance(value, bool):
            return {"enable": value}
        return value if isinstance(value, dict) else {}

    def __init__(
        self,
        config_json_path: Optional[str] = f"{HOME_DIR}/duck_config.json",
        ignore_default: bool = False,
    ):
        """
        Looks for duck_config.json in the home directory by default.
        If not found, uses default values.
        """
        self.default = False
        try:
            self.json_config = (
                json.load(open(config_json_path, "r")) if config_json_path else {}
            )
        except FileNotFoundError:
            print(
                f"Warning : didn't find the config json file at {config_json_path}, using default values"
            )
            self.json_config = {}
            self.default = True

        if config_json_path is None:
            print("Warning : didn't provide a config json path, using default values")
            self.default = True

        if self.default and not ignore_default:
            print("")
            print("")
            print("")
            print("")
            print("======")
            print(
                "WARNING : Running with default values probably won't work well. Please make a duck_config.json file and set the parameters."
            )
            res = input("Do you still want to run ? (y/N)")
            if res.lower() != "y":
                print("Exiting...")
                exit(1)

        self.start_paused = self.json_config.get("start_paused", False)
        self.imu_upside_down = self.json_config.get("imu_upside_down", False)
        self.phase_frequency_factor_offset = self.json_config.get(
            "phase_frequency_factor_offset", 0.0
        )

        expression_features = self.json_config.get("expression_features", {})

        # eyes and projector are nested sub-settings, e.g.
        #   "eyes": {"enable": false, "serial": false}
        #   "projector": {"enable": false, "serial": false, "brightness": 50}
        eyes_cfg = self._feature_config(expression_features, "eyes")
        self.eyes = eyes_cfg.get("enable", False)
        # Drive the eyes over UART serial instead of GPIO pins.
        self.eyes_serial = eyes_cfg.get("serial", False)

        projector_cfg = self._feature_config(expression_features, "projector")
        self.projector = projector_cfg.get("enable", False)
        # Drive the projector over UART serial instead of GPIO pins.
        self.projector_serial = projector_cfg.get("serial", False)
        self.projector_brightness = projector_cfg.get("brightness", 50)

        self.antennas = expression_features.get("antennas", False)
        self.speaker = expression_features.get("speaker", False)
        self.microphone = expression_features.get("microphone", False)
        self.camera = expression_features.get("camera", False)

        # default joints offsets are 0.0
        self.joints_offset = self.json_config.get(
            "joints_offsets",
            {
                "left_hip_yaw": 0.0,
                "left_hip_roll": 0.0,
                "left_hip_pitch": 0.0,
                "left_knee": 0.0,
                "left_ankle": 0.0,
                "neck_pitch": 0.0,
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "head_roll": 0.00,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.0,
                "right_hip_pitch": 0.0,
                "right_knee": 0.0,
                "right_ankle": 0.0,
            },
        )
