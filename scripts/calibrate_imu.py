
from mini_bdx_runtime.raw_imu import Imu

if __name__ == "__main__":
    print("The BNO085 performs dynamic calibration automatically in the background.")
    print("No manual calibration procedure is required.")
    print("Starting IMU to verify it reads correctly...")
    imu = Imu(50, upside_down=False)
    import time, numpy as np
    for _ in range(20):
        data = imu.get_data()
        print("gyro", np.around(data["gyro"], 3), "  accelero", np.around(data["accelero"], 3))
        time.sleep(0.1)
