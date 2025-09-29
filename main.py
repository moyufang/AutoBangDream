import subprocess
import time

from configuration import *

from utils.WinGraber import *
from utils.adb import *

#============ adb processs ============#

minitouch = Minitouch()

#============ Windows GUI ============#

SCALE, WIDTH, HEIGHT = 1, 1280, 720
full_graber = MumuGraber(SCALE, None, WIDTH, HEIGHT, 0, 0)

#============ configuration ============#
import configuration

#============ main ============#

time.sleep(1)
# minitouch.restart_game()
start_t = int((time.time()+2)*1000000+0.5)
minitouch.write(f"s {start_t} {-4629663+1000000}\n")
time.sleep(10)