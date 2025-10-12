import time
import subprocess
import os

class ADB:
  def __init__(self, commands='./commands.sheet'):
    self.open_minitouch(commands)
  def open_minitouch(self, commands='./commands.sheet'):
    os.system("adb connect 127.0.0.1:7555")
    command = ['adb', '-s', "127.0.0.1:7555", 'shell']
    self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    self.p.stdin.write(
      f'mkdir /data/local/tmp/;cd /data/local/tmp/;pkill bangcheater;./bangcheater {commands}\n'.encode())
  def write(self, s):
    if type(s) == str: self.p.stdin.write(s.encode())
    else: self.p.stdin.write(s)
    self.p.stdin.flush()
  def read(self):
    print(self.p.stdout.readline().decode())
  def close(self):
    self.p.terminate()
  def clear(self, n):
    for i in range(n):
      # self.p.stdin.write(b'd %d 100 100 10\nc\n'%(i))
      # self.p.stdin.flush()
      # time.sleep(0.01)
      self.p.stdin.write(b'u %d\nc\n'%(i))
      self.p.stdin.flush()
  def click(self, touch, pos):
    x,y = pos
    self.p.stdin.write(b'd %d %d %d 50\nc\n'%(touch, x,y))
    self.p.stdin.flush()
    time.sleep(0.05)
    self.p.stdin.write(b'u %d\nc\n'%(touch))
    self.p.stdin.flush()
  def restart_game(self):
    self.click(0, (50, 320))
    time.sleep(0.5)
    print(1)
    self.click(1, (100, 640))
    time.sleep(0.5)
    print(2)
    self.click(2, (150, 960))
    print(3)
    time.sleep(1.0)
 
def push_file(file_path:str, target_path:str='/data/local/tmp/'):
  if not os.path.exists(file_path):
    print(f"'{file_path}' does\'t exisit, drop pushing")
    return False
  os.system("adb -s 127.0.0.1:7555 push \"%s\" \"%s\""%(os.path.abspath(file_path), target_path))
  