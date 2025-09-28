import json

def refine(file_path:str):
  with open(file_path, "r", encoding='utf-8') as file:
    data = json.load(file)
  left, right = ['{', '[', '('], ['}', ']', ')']
  if isinstance(data, dict): ty = 0; items = data
  elif isinstance(data, list): ty = 1; items = range(len(data))
  elif isinstance(data, tuple): ty = 2; items = range(len(data))
  res = f"{left[ty]}\n"
  is_first = True
  for k in items:
    if not is_first: res += ",\n"; 
    else: is_first = False
    if ty == 0: res += f"\"{k}\":"+json.dumps(data[k])
    elif ty == 1 or ty == 2: res += json.dumps(data[k])
  res += f"\n{right[ty]}"
  
  
  with open(file_path, "w", encoding='utf-8') as file:
    file.write(res)
    
refine('./play/fetch_one_sheet_295_4.bestdori')