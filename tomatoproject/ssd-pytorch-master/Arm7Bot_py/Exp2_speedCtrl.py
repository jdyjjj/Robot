#!/usr/bin/python3 

# 7Bot Robotic Arm Example 2: Robot Speed Control

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

# 1. set low motion speed
arm.setSpeed(20)
pose1 = [50,  80,  50,  50,  50,  50, 40]
arm.setAngles(pose1)
time.sleep(2)

# 2. set high motion speed
arm.setSpeed(70)
pose2 = [130,  100,  80,  130,  130,  130, 80]
arm.setAngles(pose2)
time.sleep(2)

