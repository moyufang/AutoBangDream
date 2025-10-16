import json
import threading
import asyncio
import websockets as ws
import concurrent.futures as cf 
from configuration import *
from utils.log import LogE, LogD, LogI, LogS
from server.player import WinPlayer

#============ Player Configuration ============#

mumu_port = 7555
server_port = 31415
bangcheater_port = 12345

player = WinPlayer('tcp', init_scale=1)