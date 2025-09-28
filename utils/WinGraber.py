import time
import os
import win32gui, win32ui, win32con, win32api
import numpy as np
import cv2 as cv

# 通过窗口 title 找到句柄，返回窗口大小矩阵 [x1,y1,x2,y2] 以及句柄
def get_window_handle(name:str):
    win_handle = win32gui.FindWindow(None, name)
    # 获取窗口句柄
    if win_handle == 0:
        assert False, "Find not handle.\n"
        return None
    else:
        return win32gui.GetWindowRect(win_handle), win_handle

def print_all_handle():
	# 获取屏幕窗口，即所有窗口的共同祖先
	all_hd = win32gui.GetDesktopWindow()
	hd_list = []
	# 遍历子窗口
	win32gui.EnumChildWindows(all_hd, lambda hwnd,param:param.append(hwnd), hd_list)
	for handle in hd_list:
		# 输出所有窗口的句柄及其标题
		# 找到你想要的窗口，将其标题复制下来即可
		print('handle:%d title:%s'%(handle,win32gui.GetWindowText(handle)))

class Graber:
	def __init__(self, handle:int, windows_region:list, region:list = None, is_save:bool = False, path:str = None):
		
		assert win32gui.IsWindow(handle)
		self.handle = handle
		self.dc = win32gui.GetWindowDC(self.handle)
		self.sdc = win32ui.CreateDCFromHandle(self.dc)
		self.memdc = self.sdc.CreateCompatibleDC()
		
		Graber.set_window(self, windows_region)
		Graber.set_region(self, region)
		Graber.set_is_save(self, is_save, path)
		self.memdc.SelectObject(self.bmp)

	def __del__(self):
		#释放资源
		self.sdc.DeleteDC()
		self.memdc.DeleteDC()
		win32gui.ReleaseDC(self.handle, self.dc)
		del self.bmp
	def set_window(self, region:list):
		# 设置窗口的 矩形 以及 位置
		# self.region 的形式是 [x1, y1, x2, y2], 即四个非负整数组成的 list
		# 需保证 x1 < x2, y1 < y2
		# 以屏幕左上角作为原点
		# 窗口区域是 [x1,x2) x [y1, y2) 左闭右开
		# 坐标轴方向如下
		#           x  
		#  O-------->
		#  |
		#  |
		#  |
		#  |
		#y v
		assert(len(region) == 4)
		for i in range(4): assert isinstance(region[i], int)
		win32gui.MoveWindow(self.handle, region[0], region[1], region[2]-region[0], region[3]-region[1], True)
		win32gui.SetForegroundWindow(self.handle)
	def set_region(self, region:list=None):
		# 设置截屏区域
		# 默认为 handle 所指定的窗口的整个窗口
		# self.region 的形式是 [x1, y1, x2, y2], 即四个非负整数组成的 list
		# 需保证 x1 < x2, y1 < y2
		# 以 handle 所指定的窗口的左上角作为原点（非整个屏幕的左上角）
		# 截屏区域是 [x1,x2) x [y1, y2) 左闭右开
		# 坐标轴方向如下
		#           x  
		#  O-------->
		#  |
		#  |
		#  |
		#  |
		#y v
		
		if region == None:
			self.region = win32gui.GetWindowRect(self.handle)
			self.width = self.region[2]-self.region[0]
			self.height = self.region[3]-self.region[1]
			self.region = [0, 0, self.width, self.height]
		else:
			assert len(region) == 4
			for i in range(4): assert isinstance(region[i], int)
			self.width = region[2]-region[0]
			self.height = region[3]-region[1]
			self.region = region.copy()

		self.bmp = win32ui.CreateBitmap()
		self.bmp.CreateCompatibleBitmap(self.sdc, self.width, self.height)
		self.memdc.SelectObject(self.bmp)

	def set_is_save(self, is_save:bool, path:str=None):
		self.is_save = is_save
		if self.is_save:
			print(f"Bund imgs-saving path with {path}")
			assert path != None
			assert os.path.isdir(path)
			if (path[-1] not in ['/', '\\']): path += '/'
			self.path = path

	def grab_screen(self, img_name:str=None):
		# 如果保存图片的话，需要提供不包括 小数点 "." 以及 扩展名 的图片名字
		# 图片保存的格式是 "png", 文件已存在时会覆盖

		self.memdc.BitBlt((0,0), (self.width, self.height), self.sdc, (self.region[0], self.region[1]), win32con.SRCCOPY)
		arr = self.bmp.GetBitmapBits(True)
		img = np.frombuffer(arr, dtype='uint8')
		img.shape = (self.height, self.width, 4)

		# 保存图片的话涉及到磁盘存取，速度会大大降低
		if self.is_save:
			assert img_name != None
			print(f"save imgs in '{self.path+str(img_name)+'.png'}'")
			cv.imwrite(self.path+str(img_name)+'.png', img)

		return img

class MumuGraber(Graber):
  def __init__(self, scale:int=1, raw_region:list=None, raw_width:int = 1280, raw_height:int = 720, base_x:int=0, base_y:int=0):
    self.RAW_WIDTH = raw_width
    self.RAW_HEIGHT = raw_height
    # raw_region 决定了 self.SCALE 影响前的截图区域
    # raw_region 默认为 [0,0,RAW_WIDTH, RAW_HEIGHT]
    if raw_region == None:
      raw_region = [0, 0, raw_width, raw_height]
    
    #打印所有窗口的 六位句柄数字 和 窗口标题
    #print_all_handle(); exit(0)
    # 找到你需要的窗口的标题
    # 将窗口标题复制到此
    window_name = 'MuMu模拟器12'
    rect, mumu_handle = get_window_handle(window_name)
    print(f"mumu_handle:{mumu_handle} win_size:{rect}")
    # handle = win32gui.GetDesktopWindow() # 获取整个桌面的句柄
    
    self._set_window(scale, base_x, base_y)
    super().__init__(mumu_handle,self.WIN_REGION,None)
    self.set_region(raw_region)
  def _set_window(self, scale:int, base_x:int, base_y:int):
    # 经测试，mumu 模拟器会微调窗口矩形
    # 具体改变包括：
    #   在top left bottom right各加了一层 PADDING
    #   在top 再加了一层上导航栏 NAVIGATION
    # 别问为什么参数这么奇怪
    # 问就是一个像素一个像素调出来的
    self.SCALE = scale
    self.WIDTH = self.RAW_WIDTH // self.SCALE
    self.HEIGHT = self.RAW_HEIGHT // self.SCALE
    self.PADDING, self.NAVIGATION = 4, 34
    # PADDING: 模拟器四周各一层的长度
    # NAVIGATION: 模拟器上导航栏的高度
    # 这两是 mumu 模拟器在显示设置为"平板16:9"下的实验参数，比例符合均正确
    self.BASE_X = base_x
    self.BASE_Y = base_y
    self.WIN_WIDTH = self.WIDTH+2*self.PADDING
    self.WIN_HEIGHT = self.HEIGHT+self.NAVIGATION+2*self.PADDING
    self.WIN_REGION = [self.BASE_X,self.BASE_Y,self.BASE_X+self.WIN_WIDTH,self.BASE_Y+self.WIN_HEIGHT]
  def set_window(self, scale:int=None, base_x:int=None,base_y:int=None):
    if scale == None and base_x == None and base_y == None: return False
    self._set_window(scale, base_x, base_y)
    super().set_window(self.WIN_REGION)
    self.set_region(self.raw_region)
    return True
  def set_region(self, raw_region:list):
    self.raw_region = raw_region.copy()
    if raw_region == None:
      region = [0, 0, self.WIDTH, self.HEIGHT]
    else:
      assert(len(raw_region) == 4)
      for i in raw_region: assert(isinstance(i, int))
      region = [x//self.SCALE for x in raw_region]
    self.region = [self.PADDING+region[0],self.NAVIGATION+self.PADDING+region[1],self.PADDING+region[2],self.NAVIGATION+self.PADDING+region[3]]
    super().set_region(self.region)
if __name__ == "__main__":
  SCREEN_SCALE = 1 #divider
  
  graber = MumuGraber(SCREEN_SCALE, None, 640, 380, 120, 150)
  graber.set_is_save(True, './utils/')
  
  width, height, raw_region = graber.WIDTH, graber.HEIGHT, graber.raw_region
  region = [x//graber.SCALE for x in raw_region]
  time.sleep(0.5)
  graber.grab_screen("1")
  print(f"game_size:{(width, height)} grab_region:{region} raw_region:{raw_region}/{graber.RAW_WIDTH, graber.RAW_HEIGHT}\n")
  
  graber.set_region([width//4, height//4, width*3//4, height*3//4])
  time.sleep(0.5)
  graber.grab_screen("2")
  print(f"new_grab_region:{graber.raw_region}\n")
  
  graber.set_window(1, 0, 0)
  width, height, raw_region = graber.WIDTH, graber.HEIGHT, graber.raw_region
  region = [x//graber.SCALE for x in raw_region]
  time.sleep(0.5)
  graber.grab_screen("3")
  print(f"new_game_size:{(width, height)} new_grab_region:{region} newraw_region:{raw_region}/{graber.RAW_WIDTH, graber.RAW_HEIGHT}\n")
  
  