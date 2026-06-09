import board
import busio
import numpy as np
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE
from adafruit_bno08x.i2c import BNO08X_I2C

from queue import Queue
from threading import Thread
import time


class Imu:
    def __init__(
        self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True
    ):
        self.sampling_freq = sampling_freq
        self.upside_down = upside_down

        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.imu = BNO08X_I2C(i2c)

        self.imu.enable_feature(BNO_REPORT_ACCELEROMETER)
        self.imu.enable_feature(BNO_REPORT_GYROSCOPE)

        if calibrate:
            print("BNO085 calibrates automatically in the background.")
            print("No manual calibration step required.")

        self.x_offset = 0

        self.last_imu_data = {
            "gyro": [0, 0, 0],
            "accelero": [0, 0, 0],
        }
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def _remap_vector(self, v):
        # Replicates BNO055 axis_remap: swap X/Y then negate based on orientation
        x, y, z = v
        if self.upside_down:
            return np.array([-y, -x, -z])
        else:
            return np.array([-y, x, z])

    def tare_x(self):
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

            data = {
                "gyro": gyro,
                "accelero": accelero,
            }

            self.imu_queue.put(data)
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data


if __name__ == "__main__":
    imu = Imu(50, upside_down=False)
    while True:
        data = imu.get_data()
        print("gyro", np.around(data["gyro"], 3))
        print("accelero", np.around(data["accelero"], 3))
        print("---")
        time.sleep(1 / 25)
