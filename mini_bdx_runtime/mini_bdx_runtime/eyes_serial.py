from threading import Lock

import serial

# Default UART used by the eyes microcontroller.
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200

# Commands understood by the eyes microcontroller.
CMD_BLINK = "blink"
CMD_STOP = "stop"
CMD_WINK_LEFT = "wink_left"
CMD_WINK_RIGHT = "wink_right"
CMD_SET = "set"

SIDES = ("left", "right", "both")


class Eyes:
    """Drives the eyes over a UART serial link instead of GPIO pins.

    The microcontroller owns the blink loop: sending "blink <min> <max>
    <duration>" starts autonomous random blinking, and "stop" halts it. The
    other commands ("wink_left", "wink_right", "set ...") are sent on demand.
    """

    def __init__(
        self,
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        blink_duration=0.1,
        min_interval=1.0,
        max_interval=4.0,
        autostart=True,
    ):
        self.serial = serial.Serial(port, baudrate, timeout=1)

        self.blink_duration = blink_duration
        self.min_interval = min_interval
        self.max_interval = max_interval

        # Serializes writes between concurrent callers.
        self._write_lock = Lock()

        if autostart:
            self.blink()

    def _send(self, command):
        """Send a newline-terminated command over the serial link."""
        with self._write_lock:
            self.serial.write((command + "\n").encode("utf-8"))
            self.serial.flush()

    def blink(self, min_interval=None, max_interval=None, duration=None):
        """Start the MCU's autonomous blink loop.

        Defaults to the instance's intervals/duration when args are omitted.
        """
        min_interval = self.min_interval if min_interval is None else min_interval
        max_interval = self.max_interval if max_interval is None else max_interval
        duration = self.blink_duration if duration is None else duration
        self._send(f"{CMD_BLINK} {min_interval} {max_interval} {duration}")

    def wink_left(self):
        self._send(CMD_WINK_LEFT)

    def wink_right(self):
        self._send(CMD_WINK_RIGHT)

    def set_color(self, side, r, g, b, w, brightness=None):
        """Set an eye's color: "set <left|right|both> R G B W [brightness]"."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        command = f"{CMD_SET} {side} {r} {g} {b} {w}"
        if brightness is not None:
            command += f" {brightness}"
        self._send(command)

    def stop(self):
        try:
            self._send(CMD_STOP)
        finally:
            self.serial.close()


if __name__ == "__main__":
    import argparse

    # Defined inside the __main__ guard so it is not importable as a module
    # attribute; this is strictly a test entry point.
    def interactive(port, baudrate):
        """Send commands to the eyes microcontroller from stdin, one per line.

        Any typed line is sent verbatim over serial, so arbitrary commands can
        be tested. A few short aliases are expanded for convenience.
        """
        # autostart=False so random blinks don't interfere with manual testing.
        eyes = Eyes(port=port, baudrate=baudrate, autostart=False)

        aliases = {
            "l": CMD_WINK_LEFT,
            "r": CMD_WINK_RIGHT,
            "b": CMD_BLINK,
            "s": CMD_STOP,
        }

        print(f"Connected to {port} @ {baudrate} baud.")
        print("Type any command to send it verbatim. Examples:")
        print("  blink [min] [max] [duration]")
        print("  wink_left | wink_right | stop")
        print("  set <left|right|both> R G B W [brightness]")
        print("Aliases: b=blink, l=wink_left, r=wink_right, s=stop. quit/q to exit.")

        try:
            while True:
                try:
                    line = input("eyes> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                if line.lower() in ("quit", "q", "exit"):
                    break
                command = aliases.get(line.lower(), line)
                eyes._send(command)
                print(f"-> sent '{command}'")
        finally:
            eyes.stop()
            print("\nClosed serial connection.")

    parser = argparse.ArgumentParser(
        description="Interactive test for the serial eyes."
    )
    parser.add_argument("--port", default=SERIAL_PORT, help="serial port")
    parser.add_argument(
        "--baudrate", type=int, default=BAUDRATE, help="serial baudrate"
    )
    args = parser.parse_args()

    interactive(args.port, args.baudrate)
