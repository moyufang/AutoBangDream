import time
import os
import win32gui, win32ui, win32con, win32api, pyautogui
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

class Grabber:
  def __init__(
    self,
    handle        :int,          # 窗口句柄
    window_region :list,         # 窗口大小
    region        :list = None,  # 截屏区域大小，以窗口左上角为起点
    is_save       :bool = False, # 是否保存图片
    imgs_path     :str  = './'   # 存放图片的文件夹路径
  ):
    assert win32gui.IsWindow(handle)
    self.handle = handle
    self.dc     = win32gui.GetWindowDC(self.handle)
    self.sdc    = win32ui.CreateDCFromHandle(self.dc)
    self.memdc  = self.sdc.CreateCompatibleDC()
    if window_region == None: window_region = win32gui.GetWindowRect(handle)
    self.set_window(window_region, region)
    self.set_is_save(is_save, imgs_path)
    self.grab_time = -1.0
  def __del__(self):
    #释放资源
    self.sdc.DeleteDC()
    self.memdc.DeleteDC()
    win32gui.ReleaseDC(self.handle, self.dc)
    del self.bmp

  def set_is_save(self, is_save:bool, imgs_path:str = None):
    self.is_save = is_save
    if self.is_save:
      print(f"Bund imgs-saving folder path with \"{imgs_path}\"")
      assert (imgs_path != None and os.path.isdir(imgs_path))
      if (imgs_path[-1] not in ['/', '\\']): imgs_path += '/'
      self.imgs_path = imgs_path

  def set_window(self, window_region:list, region:list = None):
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
    self.window_region = window_region
    self.window_width  = window_region[2]-window_region[0]
    self.window_height = window_region[3]-window_region[1]
    self.is_set_region = False
    win32gui.MoveWindow(
      self.handle,
      window_region[0], window_region[1],
      self.window_width, self.window_height, True
    )
    # win32gui.SetForegroundWindow(self.handle)

    self.set_region(region)

  def set_region(self, region:list=None):
    # 设置截屏区域,em
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
    
    if region == None: #截取整个窗口
      self.region = self.window_region
      self.width  = self.region[2]-self.region[0]
      self.height = self.region[3]-self.region[1]
      self.region = [0, 0, self.width, self.height]
    else:
      assert len(region) == 4
      for i in range(4): assert isinstance(region[i], int)
      self.width  = region[2]-region[0]
      self.height = region[3]-region[1]
      self.region = region.copy()

    self.bmp = win32ui.CreateBitmap()
    self.bmp.CreateCompatibleBitmap(self.sdc, self.width, self.height)
    self.memdc.SelectObject(self.bmp)

  def grab(self, img_name:str=None):
    # 如果保存图片的话，需要提供不包括 小数点 "." 以及 扩展名 的图片名字
    # 图片保存的格式是 "png", 文件已存在时会覆盖

    self.memdc.BitBlt((0,0), (self.width, self.height), self.sdc, (self.region[0], self.region[1]), win32con.SRCCOPY)
    arr = self.bmp.GetBitmapBits(True)
    self.grab_time = time.time()
    img = np.frombuffer(arr, dtype='uint8')
    img.shape = (self.height, self.width, 4)

    # 如果保存图片的话，磁盘存取会显著降低 grab 速度
    if self.is_save:
      assert img_name != None
      print(f"save imgs in \"{self.imgs_path+str(img_name)+'.png'}\"")
      cv.imwrite(self.imgs_path+str(img_name)+'.png', img)

    return img

class MumuGrabber:
  def resolution2padding_navigation(resolution:tuple=(1280, 720)):
    r2p_n = {
      (16, 9): (4, 36)#(5, 58)
    }
    rate = resolution[0]/resolution[1]
    min_rate_diff = 1000000000
    result = None
    for k in r2p_n:
      x = r2p_n[k]
      if resolution[0]*x[1] == resolution[1]*x[0]: return x
      elif abs(rate-x[0]/x[1]) < min_rate_diff:
        min_rate_diff = abs(rate-x[0]/x[1])
        result = x
    return result
    
  def __init__(
    self,
    window_name     :str  = 'MuMu安卓设备',
    scale           :int  = 1,
    window_base     :list = [   0,   0],
    std_window_size :list = [1280, 720],
    std_region      :list = None,
    is_save         :bool = False,
    imgs_path       :str  = './'
  ):
    
    # 打印所有窗口的 六位句柄数字 和 窗口标题
    # print_all_handle(); exit(0)
    # 找到你需要的窗口的标题
    # 将窗口标题复制到 window_name
    rect, self.handle = get_window_handle(window_name)
    print(f"mumu_handle:{self.handle} win_size:{rect}")
    # handle = win32gui.GetDesktopWindow() # 获取整个桌面的句柄
    
    assert(len(std_window_size) == 2)
    for i in std_window_size: assert(isinstance(i, int))
    # 整个 Mumugrabber 的生命周期中，标准分辨率不可改变
    self.STD_WINDOW_WIDTH, self.STD_WINDOW_HEIGHT = std_window_size
    
    # 经测试，mumu 模拟器会微调窗口矩形
    # 具体改变包括：
    #   在top left bottom right各加了一层 PADDING
    #   在top 再加了一层上导航栏 NAVIGATION
    # PADDING、NAVIGATION 随 mumu 模拟器分辨率而改变，由经验表 r2n_p 中给出
    self.PADDING, self.NAVIGATION = \
      MumuGrabber.resolution2padding_navigation(
        (self.STD_WINDOW_WIDTH, self.STD_WINDOW_HEIGHT)
      )
    
    self.set_window(scale, window_base)
    self.set_region(std_region)
    self.grabber = Grabber(
      self.handle,
      self.__grabber_window_region,
      self.__grabber_region,
      is_save,
      imgs_path
    )
    
    self.set_is_save = lambda is_save, imgs_path=None: self.grabber.set_is_save(is_save, imgs_path)
    self.grab        = lambda img_name=None: self.grabber.grab(img_name)
    
  def set_window(self, scale:int = None, window_base:list=None, std_region:list=None):
    if scale == None and window_base == None: return False
    assert(scale == None or isinstance(scale, int))
    if window_base:
      assert(len(window_base) == 2)
      for i in window_base: assert(isinstance(i, int))
    self.scale = scale if scale else 1
    self.window_base_x, self.window_base_y = window_base if window_base else [0, 0]
    
    self.window_width  = self.STD_WINDOW_WIDTH  // self.scale
    self.window_height = self.STD_WINDOW_HEIGHT // self.scale
    
    grabber_window_width  = self.window_width  + 2*self.PADDING
    grabber_window_heigth = self.window_height + 2*self.PADDING + self.NAVIGATION
    self.__grabber_window_region = [
      self.window_base_x,
      self.window_base_y,
      self.window_base_x + grabber_window_width,
      self.window_base_y + grabber_window_heigth
    ]
    
    if hasattr(self, "grabber"):
      self.set_region(std_region if std_region else self.std_region)
      self.grabber.set_window(self.__grabber_window_region, self.__grabber_region)

    return True
    
  def set_region(self, std_region:list=None):
    if std_region == None:
      self.std_region = [0, 0, self.STD_WINDOW_WIDTH, self.STD_WINDOW_HEIGHT]
    else:
      assert(len(std_region) == 4)
      for i in std_region: assert(isinstance(i, int))
      self.std_region = std_region.copy()
    self.region = [x//self.scale for x in self.std_region]
    self.width  = self.region[2] - self.region[0]
    self.height = self.region[3] - self.region[1]
    self.__grabber_region = [
      self.PADDING    + self.region[0],
      self.NAVIGATION + self.PADDING+self.region[1],
      self.PADDING    + self.region[2],
      self.NAVIGATION + self.PADDING+self.region[3]
    ]
    if hasattr(self, 'grabber'):
      self.grabber.set_region(self.__grabber_region)
    
if __name__ == "__main__":
  SCALE, STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT = 1, 1280, 720#divider
  
  base = [0, 0]
  grabber = MumuGrabber(
    'Mumu安卓设备',
    SCALE,
    base,
    [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT],
    None
  )
  grabber.set_is_save(True, './utils/')

  time.sleep(0.5)
  grabber.grab("1")
  print(f"\
grab_region:{grabber.grabber.region}/{(grabber.grabber.window_width, grabber.grabber.window_height)} \
region:{grabber.region}/{grabber.STD_WINDOW_WIDTH, grabber.STD_WINDOW_HEIGHT}\
\n")

#   grabber.set_region([STD_WINDOW_WIDTH//4, STD_WINDOW_HEIGHT//4, STD_WINDOW_WIDTH*3//4, STD_WINDOW_HEIGHT*3//4])
#   time.sleep(0.5)
#   grabber.grab("2")
#   print(f"\
# grab_region:{grabber.grabber.region}/{(grabber.grabber.window_width, grabber.grabber.window_height)} \
# region:{grabber.region}/{grabber.STD_WINDOW_WIDTH, grabber.STD_WINDOW_HEIGHT}\
# \n")

#   grabber.set_window(2)
#   time.sleep(0.5)
#   grabber.grab("3")
#   print(f"\
# grab_region:{grabber.grabber.region}/{(grabber.grabber.window_width, grabber.grabber.window_height)} \
# region:{grabber.region}/{grabber.STD_WINDOW_WIDTH, grabber.STD_WINDOW_HEIGHT}\
# \n")
