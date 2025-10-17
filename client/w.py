import json
import threading
import asyncio
import websockets as ws
import multiprocessing as mp
from configuration import *
from utils.log import LogE, LogD, LogI, LogS
from server.controller import BangcheaterController, Controller
from server.player import WinPlayer

#============ module ============#

import asyncio
import websockets
import json
from multiprocessing import Process, Queue

from client.scriptor                import start as scriptor_start
from song_recognition.grabber       import start as add_song_start
from song_recognition.train_triplet import train as sr_train_start
from UI_recognition.add_img         import start as add_img_start
from UI_recognition.train           import train as ur_train_start
from sheet.fetch                    import start as fetch_start
from client.workflow                import start as workflow_start

class LockManager:
  def __init__(self):
    self.locks = {}
  def create_lock(self, lock:str):
    if lock in self.locks: return False
    self.locks[lock] = {"name":lock, "holder":'', "avail":True}
    return True
  def acquire(self, lock:str, applier:str='unknown'):
    if lock not in self.locks: raise ValueError("Unknown lock")
    if self.locks[lock]['avail']:
      self.locks[lock]['avail'] = False
      self.locks[lock]['holder'] = applier
      return True, applier
    else:
      return False, self.locks[lock]['holder']
  def release(self, lock:str):
    if lock not in self.locks: raise ValueError("Unknown lock")
    self.locks[lock]['avail'] = True
    self.locks[lock]['holder'] = ''
    
# 存储每个模块的进程和队列
modules = {
  'scriptor': {'process': None, 'queue': Queue(), 'start': scriptor_start, 'require_lock':'player'},
  'add_song': {'process': None, 'queue': Queue(), 'start': add_song_start, 'require_lock':'player'},
  'sr_train': {'process': None, 'queue': Queue(), 'start': sr_train_start},
  'add_img' : {'process': None, 'queue': Queue(), 'start':  add_img_start},
  'ur_train': {'process': None, 'queue': Queue(), 'start': ur_train_start},
  'fetch'   : {'process': None, 'queue': Queue(), 'start':    fetch_start},
  'workflow': {'process': None, 'queue': Queue(), 'start': workflow_start, 'require_lock':'player'}
}

def start_module(module_name, initial_config):
  queue = modules[module_name]['queue']
  process = Process(target=modules[module_name]['start'], args=(initial_config, queue))
  process.start()
  modules[module_name]['process'] = process
  
def stop_module(module_name):
  process = modules[module_name]['process']
  if process and process.is_alive():
    process.terminate()
    process.join()

# 处理WebSocket消息
async def handler(websocket, path):
  async for message in websocket:
    try:
      data = json.loads(message)
      # 期望消息格式：{"module": "module_name", "config": { ... }}
      module_name = data.get('module')
      new_config = data.get('config')
      if module_name in modules:
        # 将配置更新放入对应模块的队列
        modules[module_name]['queue'].put(new_config)
        await websocket.send(f"Config update for {module_name} sent.")
      else:
        await websocket.send(f"Unknown module: {module_name}")
    except Exception as e:
      await websocket.send(f"Error: {str(e)}")

# 启动WebSocket服务器
async def start_websocket_server():
  async with websockets.serve(handler, "localhost", WARPER_PORT):
    await asyncio.Future()  # 永远运行

if __name__ == "__main__":
  # 启动所有子任务

  # 启动WebSocket服务器
  asyncio.start(start_websocket_server())

  # 注意：由于asyncio.run会阻塞，这里我们假设在关闭程序时先停止WebSocket服务器，然后停止子任务。
  # 但实际上，我们需要处理优雅关闭。这里为了简单，先这样写。

#============ Player Configuration ============#

mumu_port = 7555
server_port = 31415
bangcheater_port = 12345

bcc = BangcheaterController("adb", f"127.0.0.1:{MUMU_PORT}", BANGCHEATER_PORT)
bcc.start_bangcheater()
clr = Controller(bcc.remote_port)
clr.connect()

