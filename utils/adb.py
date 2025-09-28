import time
import subprocess

def write(process, s):
	if type(s) == str: process.stdin.write(s.encode())
	else: process.stdin.write(s)
	process.stdin.flush()

def read(process):
	print(process.stdout.readline().decode())

def open_minitouch():
	command = ['adb', 'shell']
	process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
	return process

def close(process):
	process.terminate()

def clear(n):
	for i in range(n):
		# process.stdin.write(b'd %d 100 100 10\nc\n'%(i))
		# process.stdin.flush()
		# time.sleep(0.01)
		process.stdin.write(b'u %d\nc\n'%(i))
		process.stdin.flush()

def click(p, pos):
	global process
	x,y = pos
	process.stdin.write(b'd %d %d %d 1\nc\n'%(p, x,y))
	process.stdin.flush()
	process.stdin.write(b'u %d\nc\n'%(p))
	process.stdin.flush()
 
def restart_game():
	click(0, (333, 633))
	click(0, (333, 633))
	time.sleep(0.4)
	click(0, (325, 538))
	click(0, (325, 538))
	time.sleep(0.4)
	click(0, (313, 810))
	click(0, (313, 810))
	time.sleep(1.0) 

if __name__ == "__main__":
  clear(7)
  restart_game()
#   text="""d 1 100 0 50
# d 0 0 100 50
# m 1 90 10 50
# m 0 10 90 50
# m 0 20 80 50
# m 1 80 20 50
# m 0 20 80 50
# m 1 80 20 50
# m 0 30 70 50
# m 1 70 30 50
# m 1 60 40 50
# m 0 40 60 50
# m 0 50 50 50
# m 1 50 50 50
# u 0
# u 1"""
# for i in text.split('\n'):
#   write(process, i)
#   time.sleep(0.5)
#   print(f"i:{i}")