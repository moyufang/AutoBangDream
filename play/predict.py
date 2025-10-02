import json
from math import sqrt

with open("coefficients.json", "r", encoding='utf-8') as file:
  data = json.load(file)
  coefficient = data['v1']
  
def predict(x):
  return coefficient[0] - coefficient[1]/(x+coefficient[2])+coefficient[3]*sqrt(x+coefficient[4])