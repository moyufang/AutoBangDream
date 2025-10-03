import subprocess
import time

from configuration import *

from utils.WinGrabber import *
from utils.ADB import *

# #============ adb processs ============#

# minitouch = Minitouch()

# #============ configuration ============#
# import configuration

# #============ main ============#

# time.sleep(1)
# # minitouch.restart_game()
# start_t = int((time.time()+2)*1000000+0.5)
# minitouch.write(f"s {start_t} {-4629663+1000000}\n")
# time.sleep(10)

#============ MumuGraber ============#

full_grabber = MumuGrabber('MuMu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], None, False)
width, height = full_grabber.width, full_grabber.height