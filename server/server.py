import socket
from configuration import *
from utils.log import *
from server.player import Player

class Server:
  def __init__(self):
    self.player = Player('tcp', init_scale=1)
    self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 设置地址重用
    self.server_socket.bind(('localhost', SERVER_PORT))
  def __del__(self):
    self.server_socket.close()
  def launch(self):
    self.server_socket.listen(1)
    try:
      while True:
        LogI("Server wait connection.")
        client_socket, client_address = self.server_socket.accept() # 阻塞直到有客户端连接
        self.handle_client_connection(client_socket, client_address)
    except KeyboardInterrupt:
      LogI("Server close.")
    finally:
      self.server_socket.close()
  def handle_client_connection(self, client_socket, client_address):
    print(f"Server connect with {client_address}")
    try:
      while True:
        # 接收客户端消息
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
          LogI(f"Server disconnect with {client_address}")
          break
        rp = self.parse_data(data)
        client_socket.send(rp if rp is bytes else rp.encode('utf-8'))
    except ConnectionResetError:
      LogI(f"Client RunTimeError: {client_address}\n")
    except Exception as e:
      LogI(f"Server Error: {e}")
    finally:
      client_socket.close()
      LogI(f"Client close: {client_address}\n")
  def parse_data(self, data:str):
    str_list = data.split(' ')
    if str_list[0] == 's':
      new_scale = int(str_list[1])
      self.player.set_scale(new_scale)
      rp = str(SERVER_OK)
    elif str_list[0] == 'g':
      rp = str(self.player.get_scale())
    elif str_list[0] == 'i':
      img = self.player.full_grabber.grab()[:,:,:3]
      rp = img.tobytes()
    elif str_list[0] == 'p':
      song_duration = int(str_list[1])
      self.player.start_playing(song_duration)
      rp = str(SERVER_OK)
    elif str_list[0] == 'c':
      touch, x, y = int(str_list[1]), int(str_list[2]), int(str_list[3])
      self.player.click(touch, x, y)
      rp = str(SERVER_OK)
    else:
      rp = str(SERVER_UNKNOWN)
    return rp

if __name__ == '__main__':
  server = Server()
  server.launch()