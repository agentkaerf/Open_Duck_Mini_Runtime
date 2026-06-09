import board
import busio
import numpy as np
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR
from adafruit_bno08x.i2c import BNO08X_I2C

from queue import Queue
from threading import Thread
import time
from scipy.spatial.transform import Rotation as R


class Imu:
    def __init__(
        self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True
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

        # Rotation that replicates the BNO055 axis_remap in software.
        # upside_down=True:  out = [[-y, -x, -z]] -> matrix [[0,-1,0],[-1,0,0],[0,0,-1]]
        # upside_down=False: out = [[-y,  x,  z]] -> matrix [[0,-1,0],[ 1,0,0],[0,0, 1]]
        if upside_down:
            self._rot_remap = R.from_matrix(
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
            )
        else:
            self._rot_remap = R.from_matrix(
                np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
            )

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

            # Apply coordinate frame remap: q_new = q_remap * q_raw * q_remap_inv
            raw_rot = R.from_quat(np.array(raw_quat))
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
        print("euler (xyz)", np.around(data, 3))
        print("---")
        time.sleep(1 / 25)
