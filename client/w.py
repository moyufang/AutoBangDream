import json
import threading
import asyncio
import websockets as ws
import multiprocessing as mp
from multiprocessing import Process, Queue
from configuration import *
from module_config.scriptor_config import ScriptorConfig
from utils.log import LogE, LogD, LogI, LogS
from server.controller import BangcheaterController

#============ module ============#

from client.scriptor                import start as scriptor_start
# from song_recognition.grabber       import start as add_song_start
# from song_recognition.train_triplet import train as sr_train_start
# from UI_recognition.add_img         import start as add_img_start
# from UI_recognition.train           import train as ur_train_start
# from sheet.fetch                    import start as fetch_start
# from client.workflow                import start as workflow_start

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

class ModuleManager:
  def __init__(self):
    self.l = LockManager()
    self.l.create_lock('player')
    self.q = Queue()
    # 存储每个模块的进程和队列
    self.m = {
      'scriptor': {'P': None, 'Q': Queue(), 'logQ': Queue(), 'F': scriptor_start, 'L':'player', 'C': ScriptorConfig(SCRIPTOR_CONFIG_PATH)},
      # 'add_song': {'P': None, 'Q': Queue(), 'logQ': Queue(), 'F': add_song_start, 'L':'player'},
      # 'sr_train': {'P': None, 'Q': Queue(), 'logQ': Queue(), 'F': sr_train_start, 'L':None},
      # 'add_img' : {'P': None, 'Q': Queue(), 'logQ': Queue(), 'F':  add_img_start, 'L':None},
      # 'ur_train': {'P': None, 'Q': Queue(), 'logQ': Queue(), 'F': ur_train_start, 'L':None},
      # 'fetch'   : {'P': None, 'Q': Queue(), 'logQ': Queue(), 'F':    fetch_start, 'L':None},
      # 'workflow': {'P': None, 'Q': Queue(), 'logQ': Queue(), 'F': workflow_start, 'L':'player'}
    }

  def is_running(self, module_name):
    return self.m[module_name]['P'] is not None and self.m[module_name]['P'].is_alive()

  def start_module(self, module_name)->tuple[bool, str]:
    if self.is_running(module_name): return False, ''
    mod = self.m[module_name]
    if mod['L']: avail, holder = self.l.acquire(mod['L'])
    if avail:
      mod['P'] = Process(target=self.m[module_name]['F'], args=(self.q, mod['Q'], mod['logQ']))
      mod['P'].daemon = True
      mod['P'].start()
    return avail, holder

  def stop_module(self, module_name):
    if not self.is_running(module_name): return False
    mod = self.m[module_name]
    mod['Q'].put({'command': 'stop'})
    mod['P'].join(timeout=1)
    if mod['P'].is_alive():
      mod['P'].terminate()
      mod['P'].join()
      mod['P'] = None
    if mod['L']: self.l.release(mod['L'])
    
  def update_module_config(self, module_name:str, new_config:dict, note:dict={}):
    self.m[module_name]['C'].update(new_config, note)
    if self.is_running(module_name):
      mod = self.m[module_name]
      mod['Q'].put({'command': 'refresh_config'})
  
  def get_module_status(self, module_name):
    if not self.is_running(module_name): return {'is_running':False}
    mod = self.m[module_name]
    data = {'command': 'get_status'}
    mod['Q'].put(data)
    status = {'is_running':True}
    try: status.update(self.q.get(timeout=2))
    except Exception as e:pass
    return status
  
  def parse(self, data:dict):
    pass

if __name__ == '__main__':
  mm = ModuleManager()

  mumu_port = 7555
  server_port = 31415
  bangcheater_port = 12345

  bcc = BangcheaterController("adb", f"127.0.0.1:{MUMU_PORT}", BANGCHEATER_PORT)
  bcc.start_bangcheater()

  mm.start_module('scriptor')
  
  while True:
    while not mm.q.empty(): mm.parse(mm.q.get())
    