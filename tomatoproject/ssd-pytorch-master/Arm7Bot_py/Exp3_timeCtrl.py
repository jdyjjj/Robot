#!/usr/bin/python3 

# 7Bot Robotic Arm Example 3: Robot Motion Time Control

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

# Motion speed has priority to motion time. So set speed to maximum at frist.
arm.setSpeed(0)

# 1. set short motion time
arm.setTime(5) # 5*100ms = 500ms
pose1 = [50,  80,  50,  50,  50,  50, 40]
arm.setAngles(pose1)
time.sleep(0.5)

# 2. set long motion time
arm.setTime(30) # 30*100ms = 3000ms
pose2 = [130,  100,  80,  130,  130,  130, 80]
arm.setAngles(pose2)
time.sleep(3)

