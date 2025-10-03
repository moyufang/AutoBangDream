import json
from math import sqrt

with open("./play/coefficients.json", "r", encoding='utf-8') as file:
  data = json.load(file)
  coefficient = data['v1']
  
def predict(x):
  return coefficient[0] - coefficient[1]/(x+coefficient[2])+coefficient[3]*sqrt(x+coefficient[4])

if __name__ == "__main__":
  import numpy as np
  from matplotlib import pyplot as plt
  
  x = np.linspace(0, 1.0, 101)
  y = np.array([predict(i) for i in x])
  
  plt.plot(x, y)
  plt.show()