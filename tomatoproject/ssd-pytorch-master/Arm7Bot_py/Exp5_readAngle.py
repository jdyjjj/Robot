
# 7Bot Robotic Arm Example 5: Read Robot Pose

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

# set arm to forceless status
arm.setStatus(2)

while(True):
    # 1. read individual joint's angle
    angle_0 = arm.getAngle(0)
    angle_1 = arm.getAngle(1)
    print("angle of joint 0 is:", angle_0, "  angle of joint 1 is:", angle_1)

    # 2. read all joints' angle at once
    angles = arm.getAngles()
    print("Joints' Angles:", angles)

    time.sleep(0.3)


