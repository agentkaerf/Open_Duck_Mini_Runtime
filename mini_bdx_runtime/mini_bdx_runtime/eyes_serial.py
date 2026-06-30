import random
import time
from threading import Thread, Event, Lock

import serial

# Default UART used by the eyes microcontroller.
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200

# Commands understood by the eyes microcontroller.
CMD_BLINK = "blink"
CMD_STOP = "stop"
CMD_WINK_LEFT = "wink_left"
CMD_WINK_RIGHT = "wink_right"


class Eyes:
    """Drives the eyes over a UART serial link instead of GPIO pins.

    Mirrors the GPIO Eyes class: a background thread triggers random blinks by
    sending the "blink" command. The other commands ("wink_left", "wink_right",
    "stop") can be sent on demand.
    """

    def __init__(
        self,
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        min_interval=1.0,
        max_interval=4.0,
    ):
        self.serial = serial.Serial(port, baudrate, timeout=1)

        self.min_interval = min_interval
        self.max_interval = max_interval

        # Serializes writes between the blink thread and on-demand calls.
        self._write_lock = Lock()

        self._stop_event = Event()
        self._thread = Thread(target=self.run, daemon=True)
        self._thread.start()

    def _send(self, command):
        """Send a newline-terminated command over the serial link."""
        with self._write_lock:
            self.serial.write((command + "\n").encode("utf-8"))
            self.serial.flush()

    def blink(self):
        self._send(CMD_BLINK)

    def wink_left(self):
        self._send(CMD_WINK_LEFT)

    def wink_right(self):
        self._send(CMD_WINK_RIGHT)

    def run(self):
        try:
            while not self._stop_event.is_set():
                self.blink()
                next_blink = random.uniform(self.min_interval, self.max_interval)
                time.sleep(next_blink)
        except Exception as err:
            print(f"Error in eye thread: {err}")
            self._stop_event.set()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        try:
            self._send(CMD_STOP)
        finally:
            self.serial.close()


if __name__ == "__main__":
    e = Eyes()
    try:
        while True:
            time.sleep(1)
    finally:
        e.stop()
