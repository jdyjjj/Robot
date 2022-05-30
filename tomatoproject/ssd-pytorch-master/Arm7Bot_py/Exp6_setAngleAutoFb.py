
# 7Bot Robotic Arm Example 6: Set Pose Auto Feedback

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

# 1. set pose(angle) feedback freqency
arm.setAnglesFbFreq(10)


while(True):
    # 2. recieve new pose feedback message
    anglesFb = arm.readAnglesFb()
    print(anglesFb)



