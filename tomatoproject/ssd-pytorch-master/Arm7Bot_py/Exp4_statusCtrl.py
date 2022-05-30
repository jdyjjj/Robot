#!/usr/bin/python3 

# 7Bot Robotic Arm Example 4: Robot motor status Control

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

# 1. set motor to status 0-protection
arm.setStatus(0)
time.sleep(3)

# 2. set motor to status 1-servo
arm.setStatus(1)
time.sleep(3)

# 3. set motor to status 2-forceless
arm.setStatus(2)
time.sleep(3)


