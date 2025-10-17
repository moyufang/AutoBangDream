import json
import threading
import asyncio
import websockets as ws
import multiprocessing as mp
from configuration import *
from utils.log import LogE, LogD, LogI, LogS
from server.controller import BangcheaterController, Controller
from server.player import WinPlayer


#============ Player Configuration ============#

mumu_port = 7555
server_port = 31415
bangcheater_port = 12345

clr = LowLatencyController(
  "adb",
  f"127.0.0.1:{MUMU_PORT}",
  BANGCHEATER_PORT
)
clr.start_bangcheater()

