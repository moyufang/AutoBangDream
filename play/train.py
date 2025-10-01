import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

trace_note_path       = './play/trace_note.json'
with open(trace_note_path, "r", encoding='utf-8') as file:
  super_t_s_list = json.load(file)
  
t_t_list = []
for t_s_list in super_t_s_list:
  pass

trace_first_note_path = './play/trace_first_note.json'
with open(trace_first_note_path, "r", encoding='utf-8') as file:
  first_nost_t_s_list = json.load(file)
  
first_note_datas = [[tim, t] for tim, [t, s] in first_nost_t_s_list]

def fit_function(x, a, b, c, d, e):
  return a - b/(x + c) + d*np.sqrt(x) + e*np.log(x + 1)

def train_and_show(data_points):
  x_data = np.array([point[0] for point in data_points])
  y_data = np.array([point[1] for point in data_points])

  # 设置参数边界 (b>0, c>0, d>0, e>0)
  bounds = ([-np.inf, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

# 进行拟合
  try:
    popt, pcov = curve_fit(fit_function, x_data, y_data, bounds=bounds)
    a_fit, b_fit, c_fit, d_fit, e_fit = popt
    
    print("\n拟合结果:")
    print(f"a = {a_fit:.4f}")
    print(f"b = {b_fit:.4f} (约束: >0)")
    print(f"c = {c_fit:.4f} (约束: >0)")
    print(f"d = {d_fit:.4f} (约束: >0)")
    print(f"e = {e_fit:.4f} (约束: >0)")
    
    # 计算拟合优度
    y_pred = fit_function(x_data, *popt)
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"\n拟合优度 R² = {r_squared:.4f}")
    
  except Exception as e:
      print(f"拟合过程中出现错误: {e}")

  plt.figure(figsize=(12, 8))
  plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='数据点', s=50)
  x_fit = np.linspace(0.01, 1.0, 200)  # 从0.01开始避免除以0
  y_fit = fit_function(x_fit, a_fit, b_fit, c_fit, d_fit, e_fit)

  plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'拟合曲线\ny = {a_fit:.2f} - {b_fit:.2f}/(x + {c_fit:.2f}) + {d_fit:.2f}√x + {e_fit:.2f}·ln(x+1)')
  
  # 设置图形属性
  plt.xlabel('x', fontsize=12)
  plt.ylabel('y', fontsize=12)
  plt.title('数据散点图与拟合曲线', fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True, alpha=0.3)
  plt.xlim(0, 1.0)
  # plt.ylim(0, 10.0)

  # 显示图形
  plt.tight_layout()
  plt.show()

  # 输出拟合函数
  print(f"\n拟合函数:")
  print(f"y = {a_fit:.4f} - {b_fit:.4f}/(x + {c_fit:.4f}) + {d_fit:.4f}·√x + {e_fit:.4f}·ln(x+1)")
  
train_and_show(first_note_datas)