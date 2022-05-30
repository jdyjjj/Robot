
# 7Bot Robotic Arm Example 8: Robot IK(Inverse Kinematics) Control

# Date:    July 20th, 2020
# Author:  Jerry Peng
 
import time 
from lib.Arm7Bot import Arm7Bot


# assign serial port to 7Bot. 
# ATTENTION: Change the parameter "/dev/cu.SLAB_USBtoUART" below 
#            according to your own computer OS system. Open whatever port is 7Bot on your computer.
# Usually:  "/dev/cu.SLAB_USBtoUART" on Mac OS
#           "/dev/ttyUSB0" on Linux
#           'COM1' on Windows
arm = Arm7Bot("/dev/cu.SLAB_USBtoUART") 


# IK6 control
arm.setIK6([-50, 185, 50], [0, 0, -1])
time.sleep(1.5)
arm.setIK6([50, 185, 50], [0, 0, -1])
time.sleep(1.5)

# alarm: out of range
arm.setIK6([500, 185, 50], [0, 0, -1])
