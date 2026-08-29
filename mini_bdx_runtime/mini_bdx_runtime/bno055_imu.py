"""BNO055 IMU backend — raw gyroscope + accelerometer.

Restored from the pre-port implementation (the parent of commit f65c95f, which
replaced the BNO055 with a BNO085 in place). Kept as a separate backend so the
two chips can coexist and be selected from duck_config.json instead of one
silently inheriting the other's constants — which is exactly how the 180 deg
yaw error documented in imu_axis_remap.py came about.

Two behavioural differences from the BNO085 backend, both intrinsic to the chip:

  * The remap is applied ON-CHIP via the BNO055's `axis_remap` register rather
    than in software. The same config spec drives both backends; this one
    translates it to the hardware 6-tuple.
  * Calibration is EXPLICIT and persisted. The BNO055 must be calibrated (see
    `calibrate=True`, which walks the calibration and writes imu_calib_data.pkl)
    and those offsets are restored on subsequent boots. This is more work than
    the BNO085's background self-calibration, but it is deterministic across
    sessions, which the BNO085's is not.
"""

import os
import pickle
import time
from queue import Queue
from threading import Thread

import adafruit_bno055
import board
import busio
import numpy as np

from mini_bdx_runtime.imu_axis_remap import AxisRemap, resolve

CALIB_PATH = "imu_calib_data.pkl"


# TODO filter spikes
class Bno055Imu:
    def __init__(
        self,
        sampling_freq,
        user_pitch_bias=0,
        calibrate=False,
        upside_down=True,
        axis_remap=None,
    ):
        self.sampling_freq = sampling_freq
        self.calibrate = calibrate
        self.upside_down = upside_down
        self.remap = (
            axis_remap
            if isinstance(axis_remap, AxisRemap)
            else resolve("bno055", upside_down, axis_remap)
        )
        print(f"[IMU] BNO055, axis remap {self.remap} (applied on-chip)")

        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_bno055.BNO055_I2C(i2c)
        self.imu.mode = adafruit_bno055.NDOF_MODE

        # Same spec as the BNO085 backend, expressed in the chip's own format.
        self.imu.axis_remap = self.remap.to_bno055_axis_remap(adafruit_bno055)

        if self.calibrate:
            self._run_calibration()

        self._restore_calibration()

        self.x_offset = 0  # see Bno085Imu.tare_x caveat before using tare_x()

        self.last_imu_data = {"gyro": [0, 0, 0], "accelero": [0, 0, 0]}
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def _run_calibration(self):
        """Interactively calibrate, save offsets, then exit (as before)."""
        self.imu.mode = adafruit_bno055.NDOF_MODE
        while not self.imu.calibrated:
            print("Calibration status: ", self.imu.calibration_status)
            time.sleep(0.1)
        print("CALIBRATION DONE")

        imu_calib_data = {
            "offsets_accelerometer": self.imu.offsets_accelerometer,
            "offsets_gyroscope": self.imu.offsets_gyroscope,
            "offsets_magnetometer": self.imu.offsets_magnetometer,
        }
        for k, v in imu_calib_data.items():
            print(k, v)

        pickle.dump(imu_calib_data, open(CALIB_PATH, "wb"))
        print("Saved", CALIB_PATH)
        exit()

    def _restore_calibration(self):
        if not os.path.exists(CALIB_PATH):
            print(f"{CALIB_PATH} not found")
            print("Imu is running uncalibrated")
            return

        imu_calib_data = pickle.load(open(CALIB_PATH, "rb"))
        self.imu.mode = adafruit_bno055.CONFIG_MODE
        time.sleep(0.1)
        self.imu.offsets_accelerometer = imu_calib_data["offsets_accelerometer"]
        self.imu.offsets_gyroscope = imu_calib_data["offsets_gyroscope"]
        self.imu.offsets_magnetometer = imu_calib_data["offsets_magnetometer"]
        self.imu.mode = adafruit_bno055.NDOF_MODE
        time.sleep(0.1)

    def tare_x(self):
        """Zero accel X. Read the caveat in Bno085Imu.__init__ before using."""
        print("Taring x ...")
        x_values = []
        num_values = 100
        ok = False
        while not ok:
            x_values.append(np.array(self.imu.acceleration)[0])
            x_values = x_values[-num_values:]

            if len(x_values) == num_values:
                mean = np.mean(x_values)
                std = np.std(x_values)
                if std < 0.05:
                    ok = True
                    self.x_offset = mean
                    print("Tare x done")
                else:
                    print(std)

            time.sleep(0.01)

    def imu_worker(self):
        while True:
            s = time.time()
            try:
                # Already in body frame: the remap is applied on-chip.
                gyro = np.array(self.imu.gyro).copy()
                accelero = np.array(self.imu.acceleration).copy()
            except Exception as e:
                print("[IMU]:", e)
                continue

            if gyro is None or accelero is None:
                continue

            if gyro.any() is None or accelero.any() is None:
                continue

            accelero[0] -= self.x_offset

            data = {"gyro": gyro, "accelero": accelero}

            self.imu_queue.put(data)
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data
