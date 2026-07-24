from threading import Lock

import serial

# Default UART used by the projector microcontroller.
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200

# Command understood by the projector microcontroller.
CMD_FLASHLIGHT = "flashlight"

DEFAULT_BRIGHTNESS = 50


class Projector:
    """Drives the projector/flashlight over a UART serial link.

    Sends "flashlight <0-100>" commands. switch() toggles between off (0) and
    the configured brightness, matching the GPIO Projector's API.
    """

    def __init__(
        self,
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        brightness=DEFAULT_BRIGHTNESS,
        autostart=False,
    ):
        self.serial = serial.Serial(port, baudrate, timeout=1)

        self.brightness = self._clamp(brightness)
        self.on = False

        # Serializes writes between concurrent callers.
        self._write_lock = Lock()

        if autostart:
            self.switch()

    @staticmethod
    def _clamp(level):
        return max(0, min(100, int(level)))

    def _send(self, command):
        """Send a newline-terminated command over the serial link."""
        with self._write_lock:
            self.serial.write((command + "\n").encode("utf-8"))
            self.serial.flush()

    def set_flashlight(self, level):
        """Set the flashlight brightness (0-100)."""
        self._send(f"{CMD_FLASHLIGHT} {self._clamp(level)}")

    def switch(self):
        """Toggle between off and the configured brightness."""
        self.on = not self.on
        self.set_flashlight(self.brightness if self.on else 0)

    def stop(self):
        try:
            self.set_flashlight(0)
        finally:
            self.serial.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the serial projector by setting a flashlight level."
    )
    parser.add_argument("--port", default=SERIAL_PORT, help="serial port")
    parser.add_argument(
        "--baudrate", type=int, default=BAUDRATE, help="serial baudrate"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_BRIGHTNESS,
        help="flashlight level to send (0-100)",
    )
    args = parser.parse_args()

    p = Projector(port=args.port, baudrate=args.baudrate)
    p.set_flashlight(args.level)
    print(f"Sent 'flashlight {Projector._clamp(args.level)}' to {args.port}")
    p.serial.close()
