import subprocess
import time

from configuration import *

from utils.WinGraber import *
from utils.adb import *

#============ adb processs ============#

process = open_minitouch()
write(process, 'mkdir /data/local/tmp/;cd /data/local/tmp/;pkill bandcheater;./bandcheater -i\n')
clear()

#============ Windows GUI ============#

SCALE, WIDTH, HEIGHT = 1, 1280, 720
full_graber = MumuGraber(SCALE, None, WIDTH, HEIGHT, 0, 0)

#============ configuration ============#
import configuration

#============ main ============#



