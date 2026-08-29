import board
import busio
import numpy as np
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR
from adafruit_bno08x.i2c import BNO08X_I2C

from queue import Queue
from threading import Thread
import time
from scipy.spatial.transform import Rotation as R

from mini_bdx_runtime.imu_axis_remap import AxisRemap, resolve


class Imu:
    """BNO085 in quaternion (game rotation vector) mode.

    Separate from the raw gyro/accel backends in bno085_imu.py: this one is used
    by imu_server.py, not by the walk runtime. It shares their configurable axis
    remap (see imu_axis_remap.py) so the two cannot drift apart — they describe
    the same physical chip on the same physical mounting.
    """

    def __init__(
        self,
        sampling_freq,
        user_pitch_bias=0,
        calibrate=False,
        upside_down=True,
        axis_remap=None,
        chip="bno085",
    ):
        self.sampling_freq = sampling_freq
        self.user_pitch_bias = user_pitch_bias
        self.nominal_pitch_bias = 0
        self.upside_down = upside_down

        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.imu = BNO08X_I2C(i2c)

        # GAME_ROTATION_VECTOR: 9DOF without magnetometer (equivalent to IMUPLUS_MODE)
        # Use BNO_REPORT_ROTATION_VECTOR for absolute orientation with magnetometer
        self.imu.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)

        if calibrate:
            print("BNO085 calibrates automatically in the background.")
            print("No manual calibration step required.")

        self.pitch_bias = self.nominal_pitch_bias + self.user_pitch_bias

        # Chip -> body axis mapping, from configuration rather than hardcoded.
        # AxisRemap guarantees a proper rotation (det=+1), which R.from_matrix
        # requires anyway; a reflection would raise there with a far less
        # helpful message.
        self.remap = (
            axis_remap
            if isinstance(axis_remap, AxisRemap)
            else resolve(chip, upside_down, axis_remap)
        )
        print(f"[IMU] {chip} (quaternion mode), axis remap {self.remap}")
        self._rot_remap = R.from_matrix(self.remap.matrix)

        self.last_imu_data = [0, 0, 0, 0]
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def imu_worker(self):
        while True:
            s = time.time()
            try:
                # BNO085 returns (i, j, k, real) = (x, y, z, w) — scalar last
                raw_quat = self.imu.game_quaternion
            except Exception as e:
                print("[IMU]:", e)
                continue

            if raw_quat is None:
                continue

            q = np.array(raw_quat)
            if np.linalg.norm(q) < 1e-6:
                continue

            # Apply coordinate frame remap: q_new = q_remap * q_raw * q_remap_inv
            raw_rot = R.from_quat(q)
            remapped_rot = self._rot_remap * raw_rot * self._rot_remap.inv()

            euler = remapped_rot.as_euler("xyz")
            euler[1] -= np.deg2rad(self.pitch_bias)

            final_orientation_quat = R.from_euler("xyz", euler).as_quat()

            self.imu_queue.put(final_orientation_quat.copy())
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self, euler=False, mat=False):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        try:
            if not euler and not mat:
                return self.last_imu_data
            elif euler:
                return R.from_quat(self.last_imu_data).as_euler("xyz")
            elif mat:
                return R.from_quat(self.last_imu_data).as_matrix()
        except Exception as e:
            print("[IMU]: ", e)
            return None


if __name__ == "__main__":
    imu = Imu(50, upside_down=False)
    while True:
        data = imu.get_data(euler=True)
        if data is not None:
            print("euler (xyz)", np.around(data, 3))
        time.sleep(1 / 25)
