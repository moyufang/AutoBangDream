import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib

colorful_lines_path   = './play/colorful_lines.png'
compare_result_path   = './play/compare_result.png'

def fit_function(x, a, b, c, d, e):
  return a - b/(x + c) + d*np.sqrt(x+e)

def train(data_points, is_show:bool=True, maxfev:int = 600):
  x_data = np.array([point[0] for point in data_points])
  y_data = np.array([point[1] for point in data_points])
  # print(f"x_data: {x_data}\ny_data: {y_data}\n")

  bounds = ([-np.inf, 0, 0, 0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf])

  popt, pcov = curve_fit(fit_function, x_data, y_data, bounds=bounds, maxfev=maxfev)
  a_fit, b_fit, c_fit, d_fit, e_fit = popt
  
  print("\n拟合结果:")
  print(f"a = {a_fit:.4f} {b_fit:.4f} {c_fit:.4f} {d_fit:.4f} {e_fit:.4f}")
  formula = f'y = {a_fit:.4f} - {b_fit:.4f}/(x + {c_fit:.4f}) + {d_fit:.4f}·√(x+{e_fit:.4f})'
  print(formula)
  
  # 计算拟合优度
  y_pred = fit_function(x_data, *popt)
  ss_res = np.sum((y_data - y_pred)**2)
  ss_tot = np.sum((y_data - np.mean(y_data))**2)
  r_squared = 1 - (ss_res / ss_tot)
  print(f"\n拟合优度 R² = {r_squared:.4f}")

  if is_show:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(12, 8))
    plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='数据点', s=1)
    x_fit = np.linspace(0.01, 1.0, 200)  # 从0.01开始避免除以0
    y_fit = fit_function(x_fit, a_fit, b_fit, c_fit, d_fit, e_fit)
    
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'拟合曲线\n'+formula)
    
    # 设置图形属性
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Track', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.show()
  
  return [a_fit, b_fit, c_fit, d_fit, e_fit], r_squared

def merge(note_datas, is_show:bool=True):
  
  note_datas = sorted(note_datas, key=lambda p: (p[1], p[0]))
  dy, dx = 0.03, 1.8
  
  lx, ly = -100, -10
  lines = []
  line = []
  for [x,y] in note_datas:
    dr = (x-lx)**2/dx**2+(y-ly)**2/dy**2
    if dr <= 1:
      line.append([x, y])
    else:
      if line != []: lines.append(line)
      line = []
      line.append([x, y])
    lx, ly = x, y
  if line != []: lines.append(line)
  
  if is_show:
    color = 0
    color_list = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown']
    if is_show: plt.figure(figsize=(1, 130))
    for line in lines:
      x_data = [p[0] for p in line]
      y_data = [p[1] for p in line]
      plt.scatter(x_data, y_data, color=color_list[color], alpha=0.6, label='数据点', s=1)
      color = (color+1)%len(color_list)
    plt.xlim(0, 1.0)
    plt.ylim(0, 130)
    plt.savefig(colorful_lines_path, dpi=300, bbox_inches='tight')
  
  merge_datas = []
  for i in range(len(lines)):
    n, mean_y = len(lines[i]), 0.0
    for j in range(n): mean_y += lines[i][j][1]
    mean_y /= n
    for j in range(n):
      x = lines[i][j][0]
      merge_datas.append([x, lines[i][j][1] - mean_y + fit_function(x, *coefficients_v1)])
    
  return merge_datas
    
def compare(first_note_datas, merge_datas, coefficients_v1, coefficients_v2):

  matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
  
  plt.figure(figsize=(12, 8))
  
  formula_v1 = 'y = %.4f - %.4f/(x + %.4f) + %.4f·√(x+%.4f)'%tuple(coefficients_v1)
  formula_v2 = 'y = %.4f - %.4f/(x + %.4f) + %.4f·√(x+%.4f)'%tuple(coefficients_v2)

  x_data = np.array([p[0] for p in merge_datas])
  y_data = np.array([p[1] for p in merge_datas])
  
  plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='数据点', s=1)
  x_fit = np.linspace(0.01, 1.0, 200)  # 从0.01开始避免除以0
  y_fit_v1 = fit_function(x_fit, *coefficients_v1)
  y_fit_v2 = fit_function(x_fit, *coefficients_v2)
  
  r2_diff_v11 = r2_diff_v1
  r2_diff_v22 = r2_diff_v2
  
  y_pred = fit_function(x_data, *coefficients_v1)
  ss_res = np.sum((y_data - y_pred)**2)
  ss_tot = np.sum((y_data - np.mean(y_data))**2)
  r2_diff_v12 = 1 - (ss_res / ss_tot)
  
  x_data = np.array([p[0] for p in first_note_datas])
  y_data = np.array([p[1] for p in first_note_datas])
  y_pred = fit_function(x_data, *coefficients_v2)
  ss_res = np.sum((y_data - y_pred)**2)
  ss_tot = np.sum((y_data - np.mean(y_data))**2)
  r2_diff_v21 = 1 - (ss_res / ss_tot)

  plt.plot(x_fit, y_fit_v1, 'r-', linewidth=2, label=f'初次曲线\nR1={r2_diff_v11} R2={r2_diff_v12}\n'+formula_v1)
  plt.plot(x_fit, y_fit_v2, 'y-', linewidth=2, label=f'重整曲线\nR1={r2_diff_v21} R2={r2_diff_v22}\n'+formula_v2)
  
  # 设置图形属性
  plt.xlabel('x', fontsize=12)
  plt.ylabel('y', fontsize=12)
  plt.xlim(0, 1.0)
  plt.tight_layout()
  plt.savefig(compare_result_path, dpi=300, bbox_inches='tight')
  plt.show()

if __name__ == "__main__":
  coefficients_path     = './play/coefficients.json'
  trace_first_note_path = './play/trace_first_note.json'
  trace_note_path       = './play/trace_note.json'
  with open(trace_first_note_path, "r", encoding='utf-8') as file: first_note_t_s_list = json.load(file)
  with open(trace_note_path, "r", encoding='utf-8') as file: super_t_s_list = json.load(file)
  
  first_note_datas = [[t, tim] for tim, [t, s] in first_note_t_s_list]
  coefficients_v1, r2_diff_v1 = train(first_note_datas, False)

  note_datas = []
  for tim, t_s_list in super_t_s_list:
    for [t,s] in t_s_list: note_datas.append([t, tim-fit_function(t, *coefficients_v1)])

  merge_datas = merge(note_datas, False)
  coefficients_v2, r2_diff_v2 = train(merge_datas, False, 1000)

  compare(first_note_datas, merge_datas, coefficients_v1, coefficients_v2)

  base_v1 = fit_function(1, *coefficients_v1)
  base_v2 = fit_function(1, *coefficients_v2)
  coefficients_v1[0] -= base_v1
  coefficients_v2[0] -= base_v2
  for i in [0, 1, 3]:
    coefficients_v1[i] *= -1
    coefficients_v2[i] *= -1
  with open(coefficients_path, 'w', encoding='utf-8') as file:
    json.dump({
      "v1": coefficients_v1,
      "v2": coefficients_v2
    }, file)
