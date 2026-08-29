# Open Duck Mini Runtime

## Raspberry Pi zero 2W setup

### Install Raspberry Pi OS

Download Raspberry Pi OS Lite (64-bit) from here : https://www.raspberrypi.com/software/operating-systems/

Follow the instructions here to install the OS on the SD card : https://www.raspberrypi.com/documentation/computers/getting-started.html

With the Raspberry Pi Imager, you can pre-configure session, wifi and ssh. Do it like below :

![imager_setup](https://github.com/user-attachments/assets/7a4987b2-de83-41dd-ab7f-585259685f16)

> Tip: I configure the rasp to connect to my phone's hotspot, this way I can connect to it from anywhere.

### Setup SSH (If not setup during the installation)

When first booting on the rasp, you will need to connect a screen and a keyboard. The first thing you should do is connect to a wifi network and enable SSH.

To do so, you can follow this guide : https://www.raspberrypi.com/documentation/computers/configuration.html#setting-up-wifi

Then, you can connect to your rasp using SSH without having to plug a screen and a keyboard.

### Update the system and install necessary stuff

```bash
sudo apt update
sudo apt upgrade
sudo apt install git
sudo apt install python3-pip
sudo apt install python3-virtualenvwrapper
(optional) sudo apt install python3-picamzero

```

Add this to the end of the `.bashrc`:

```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
```

### Enable I2C

`sudo raspi-config` -> `Interface Options` -> `I2C`

Set the I2C bus to 400 kHz

```bash
sudo nano /boot/firmware/config.txt
# copy the following line to /boot/firmware/config.txt
dtparam=i2c_arm_baudrate=400000
```

Reboot for settings to take effect

### Set the usbserial latency timer

```bash
cd  /etc/udev/rules.d/
sudo touch 99-usb-serial.rules
sudo nano 99-usb-serial.rules
# copy the following line in the file
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
```

### Set the udev rules for the motor control board

TODO


### Setup xbox one controller over bluetooth

Turn your xbox one controller on and set it in pairing mode by long pressing the sync button on the top of the controller.

Run the following commands on the rasp :

```bash
bluetoothctl
scan on
```

Wait for the controller to appear in the list, then run :

```bash
pair <controller_mac_address>
trust <controller_mac_address>
connect <controller_mac_address>
```

The led on the controller should stop blinking and stay on.

You can test that it's working by running

```bash
python3 mini_bdx_runtime/mini_bdx_runtime/xbox_controller.py
```

## Speaker wiring and configuration
Follow this tutorial

> For now, don't activate `/dev/zero` when they ask

https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp?view=all


## Install the runtime

### Make a virtual environment and activate it

```bash
mkvirtualenv -p python3 open-duck-mini-runtime
workon open-duck-mini-runtime
```

Clone this repository on your rasp, cd into the repo, then :

```bash
git clone https://github.com/apirrone/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime
git checkout v2
pip install -e .
```

In Raspberry Pi 5, you need to perform the following operations

```bash
pip uninstall -y RPi.GPIO
pip install lgpio
```


## Test the IMU

```bash
python3 mini_bdx_runtime/mini_bdx_runtime/raw_imu.py
```

You can also run `python3 scripts/imu_server.py` on the robot and `python3 scripts/imu_client.py --ip <robot_ip>` on your computer to check that the frame is oriented correctly. 

> To find the ip address of the robot, run `ifconfig` on the robot

### Verify the IMU orientation (do this before trusting any policy)

**A wrong IMU axis mapping is invisible in simulation and will make the robot
unstable on hardware.** The simulated IMU is perfectly aligned, so every sim
metric can look fine while the real robot fights itself. If the mapping has a
sign error on X or Y, balance corrections are applied in the *wrong direction* —
positive feedback — which looks like the robot rocking with growing amplitude
until it tips, or falling in whichever direction it is walking.

Run the diagnostic battery:

```bash
python3 scripts/diagnose_imu.py               # all tests
python3 scripts/diagnose_imu.py --test axes   # just the 6-position axis test
```

It runs three checks, ordered by how few assumptions they make:

1. **Invariants** (pose-independent) — `|accel|` must be ~9.81 m/s² and gyro ~0
   in *any* stationary orientation. Valid even hand-held, so it does not depend
   on getting the robot into a particular pose.
2. **Axis / remap** (6-position test) — hold the robot with each body axis up
   and down in turn; confirms gravity lands on the expected axis with the
   expected sign. This is what validates your `imu_axis_remap`.
3. **Standing bias** (pose-specific) — with the robot free-standing in its home
   pose on level ground, untouched, compares against the reference vector from
   simulation. Deviation on X reads as phantom pitch at roughly 5.8° per 1 m/s².
   Fix the axis mapping first — a remap error masquerades as a bias, and often
   accounts for all of it. Note there is currently **no working correction knob**
   for a residual bias on this path: `--pitch_bias` is a no-op in the raw
   gyro/accel backend the walk runtime uses (only the quaternion backend applies
   it), and `tare_x()` must not be used because it zeroes a legitimate gravity
   component of the home pose.

To try a candidate mapping without editing the config first:

```bash
python3 scripts/diagnose_imu.py --test axes --axis-remap "y,x,-z"
```

> The BNO085 self-calibrates continuously (the BNO055 used stored calibration
> instead), so its bias is **not** deterministic across sessions. Use
> `--log imu_runs.jsonl` and repeat across several power cycles before
> concluding anything about a systematic offset.

### IMU configuration

Three keys in `duck_config.json` describe the IMU:

| key | values | meaning |
|---|---|---|
| `imu_chip` | `"bno085"` (default), `"bno055"` | Which IMU is physically fitted. Each has its own backend module. |
| `imu_upside_down` | `true` / `false` (default `false`) | Whether the board is mounted inverted. |
| `imu_axis_remap` | `null` (default), a spec, or a preset name | Optional explicit chip→body axis mapping. Overrides the default implied by `imu_chip` + `imu_upside_down`. |

Leave `imu_axis_remap` as `null` unless the 6-position test fails — the chip and
mounting already select a sensible default.

**Spec format.** Three comma-separated terms giving the source of body X, Y, Z
(body frame is x=forward, y=left, z=up):

```
"y,x,-z"   ->   body_x = +chip_y,   body_y = +chip_x,   body_z = -chip_z
```

Any permutation of `x`/`y`/`z` with independent signs is accepted, provided it
is a **proper rotation**. A mapping with determinant −1 (a mirror) is rejected
at startup: it would transform the accelerometer correctly while silently
inverting the gyro, since angular velocity is an axial vector and acceleration
is a polar one. That failure is nearly impossible to spot by eye.

Preset names accepted in place of a spec:

| preset | spec | notes |
|---|---|---|
| `bno085_upside_down` | `y,x,-z` | Verified on hardware (6/6 positions) |
| `bno085_normal` | `-y,x,z` | Inherited, **not** hardware-verified — run the axis test |
| `bno055_upside_down` | `-y,-x,-z` | Matches the original BNO055 hardware `axis_remap` |
| `bno055_normal` | `-y,x,z` | Matches the original BNO055 hardware `axis_remap` |
| `identity` | `x,y,z` | No remap; chip frame == body frame |

On the BNO055 the same spec is applied by the chip itself via its `axis_remap`
register; on the BNO085 it is applied in software. Either way the meaning is
identical, so a mapping verified on one is expressed the same way on the other.

## Test motors

This will allow you to verify all your motors are connected and configured.

```bash
python3 scripts/check_motors.py
```

## Make your duck_config.json

Copy `example_config.json` in the home directory of your duck and rename it `duck_config.json`.

`cp example_config.json ~/duck_config.json`

In this file, you can configure some stuff, like registering if you installed the expression features, which IMU you fitted and how it is mounted (`imu_chip`, `imu_upside_down`, `imu_axis_remap` — see [IMU configuration](#imu-configuration)) and other stuff. You also write the joints offsets of your duck here

## Find the joints offsets

This script will guide you through finding the joints offsets of your robot that you can then write in your `duck_config.json`

> This procedure won't be necessary in the future as we will be flashing the offsets directly in each motor's eeprom.

```bash
cd scripts/
python find_soft_offsets.py
```

## Run the walk !

Download the [latest policy checkpoint ](https://github.com/apirrone/Open_Duck_Mini/blob/v2/BEST_WALK_ONNX_2.onnx) and copy it to your duck.

`cd scripts/`

`python v2_rl_walk_mujoco.py --onnx_model_path <path_to>/BEST_WALK_ONNX_2.onnx`



| Control | Action |
|---|---|
| Left stick | Linear velocity (forward/back, strafe) |
| Right stick X | Angular (yaw) velocity |
| A | Pause / unpause |
| B | Play a random sound |
| X | Toggle projector |
| Y | Toggle head control mode (left stick controls head joints instead of body velocity. EXPERIMENTAL - can break your head!) |
| LB (hold) | Sprint — increases gait frequency |
| D-pad up / down | Increase / decrease base gait cadence |
| Left trigger | Right antenna position |
| Right trigger | Left antenna position |

### Head control mode (`--head_mode`)

When head control mode is toggled with the Y button, the manual head command can still affect the legs (the policy keeps stepping and reacts to the head command in its observation). The `--head_mode` option controls how the legs are isolated while head control is active:

| Value | Gait | Head command in policy obs | Effect |
|---|---|---|---|
| `none` (default) | runs | fed in | Original behavior — the legs still react to the head stick |
| `freeze` | paused, legs held | fed in | Robot stands still, only the head moves |
| `decouple` | runs | hidden | Legs keep balancing but ignore the head stick |
| `both` | paused, legs held | hidden | Strongest isolation: no leg reaction and no gait drift |

Example:

`python v2_rl_walk_mujoco.py --onnx_model_path <path_to>/BEST_WALK_ONNX_2.onnx --head_mode both`

Note: `freeze` and `both` pause the policy's active leg balancing while head control is held, so the robot holds a static stance. If your duck tends to tip when standing still, prefer `decouple`.