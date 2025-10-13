from server.player import Player, PlayerServer
from configuration import *
from utils.log import *

if __name__ == '__main__':
  server = PlayerServer()
  server.launch()