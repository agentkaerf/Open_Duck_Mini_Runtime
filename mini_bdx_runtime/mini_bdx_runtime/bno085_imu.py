"""BNO085 IMU backend — raw gyroscope + accelerometer.

Reports gyro (rad/s) and accelerometer (m/s^2, gravity included) already mapped
into the robot body frame, which is what the trained policy's observations
expect. The axis mapping is configuration-driven; see imu_axis_remap.py.

Calibration note (differs from the BNO055): the BNO085 self-calibrates
continuously in the background rather than using stored calibration data. That
is convenient but has a consequence worth remembering — its bias is NOT
deterministic across sessions, and accelerometer bias estimation in particular
needs the chip to see multiple orientations before it converges. A robot that
lives upright may never fully converge. If you are chasing a systematic offset,
measure it across several power cycles (scripts/diagnose_imu.py --log) rather
than trusting a single reading.
"""

import time
from queue import Queue
from threading import Thread

import board
import busio
import numpy as np
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE
from adafruit_bno08x.i2c import BNO08X_I2C

from mini_bdx_runtime.imu_axis_remap import AxisRemap, resolve


class Bno085Imu:
    def __init__(
        self,
        sampling_freq,
        user_pitch_bias=0,
        calibrate=False,
        upside_down=True,
        axis_remap=None,
    ):
        self.sampling_freq = sampling_freq
        self.upside_down = upside_down
        self.remap = (
            axis_remap
            if isinstance(axis_remap, AxisRemap)
            else resolve("bno085", upside_down, axis_remap)
        )
        print(f"[IMU] BNO085, axis remap {self.remap}")

        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self._init_imu()

        if calibrate:
            print("BNO085 calibrates automatically in the background.")
            print("No manual calibration step required.")

        # NOTE: `user_pitch_bias` is accepted for signature compatibility but is
        # NOT APPLIED in this raw gyro/accel backend — it never has been, in
        # this or the pre-refactor raw_imu.py, or the BNO055 version before it
        # (where the line was present but commented out). Only imu.py, the
        # quaternion backend behind imu_server.py, honours it. That means
        # `v2_rl_walk_mujoco.py --pitch_bias` currently does nothing, since the
        # walk runtime uses this backend. If a genuine standing bias survives a
        # verified-correct axis remap, pitch correction has to be implemented
        # here (rotate the remapped accel/gyro about body Y) before that flag
        # means anything.
        self.user_pitch_bias = user_pitch_bias

        # Accelerometer X bias correction, applied after the remap.
        # NOTE: tare_x() forces body-frame accel X to read ZERO, which is only
        # valid in a pose whose true X component is genuinely zero. In the
        # robot's standing home pose the true value is about -0.372 m/s^2 (a
        # real gravity component of the trunk's pitch), so taring there deletes
        # signal and manufactures a ~2.2 deg phantom lean.
        self.x_offset = 0

        self.last_imu_data = {"gyro": [0, 0, 0], "accelero": [0, 0, 0]}
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def _init_imu(self, retries=10):
        """Initialize the BNO085 and enable features.

        The BNO08x is flaky over I2C (clock-stretching issues on the Pi), so
        enabling features can fail intermittently. Retry a full re-init until
        both features enable cleanly.
        """
        last_err = None
        for attempt in range(retries):
            try:
                self.imu = BNO08X_I2C(self.i2c)
                # Give the chip a moment to finish booting after (re)init.
                time.sleep(0.5)
                self.imu.enable_feature(BNO_REPORT_ACCELEROMETER)
                self.imu.enable_feature(BNO_REPORT_GYROSCOPE)
                return
            except Exception as e:
                last_err = e
                print(f"[IMU]: init attempt {attempt + 1}/{retries} failed: {e}")
                time.sleep(0.5)

        raise RuntimeError(
            f"Failed to initialize BNO085 after {retries} attempts"
        ) from last_err

    def _remap_vector(self, v):
        return self.remap.apply(v)

    def tare_x(self):
        """Zero the body-frame accel X. See the caveat in __init__ first."""
        print("Taring x ...")
        x_values = []
        num_values = 100
        ok = False
        while not ok:
            raw = self.imu.acceleration
            if raw is None:
                time.sleep(0.01)
                continue
            x_values.append(self._remap_vector(np.array(raw))[0])
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
                raw_gyro = self.imu.gyro
                raw_accelero = self.imu.acceleration
            except Exception as e:
                print("[IMU]:", e)
                continue

            if raw_gyro is None or raw_accelero is None:
                continue

            gyro = self._remap_vector(np.array(raw_gyro))
            accelero = self._remap_vector(np.array(raw_accelero))
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
