import json

def refine(file_path:str, max_depth=[2, 0, 0], indent='  '):
  with open(file_path, "r", encoding='utf-8') as file:
    data = json.load(file)
  left, right = ['{', '[', '('], ['}', ']', ')']
  res = ''
  
  def rf(data:dict, key:any, depth = 0):
    nonlocal res
    
    is_object = False
    if isinstance(data, dict): ty = 0; items = data
    elif isinstance(data, list): ty = 1; items = range(len(data))
    elif isinstance(data, tuple): ty = 2; items = range(len(data))
    else: ty = 4; is_object = True
    
    prefix = f"\"{key}\":" if key is not None else ''
    if is_object or depth >= max_depth[ty]:
      res += indent*depth + prefix + json.dumps(data)
    else:
      res += indent*depth + prefix + f"{left[ty]}\n"
      is_first = True
      for k in items:
        if not is_first: res += ",\n"; 
        else: is_first = False
        if ty == 0: rf(data[k], key=k, depth=depth+1)
        elif ty == 1 or ty == 2: rf(data[k], key=None, depth=depth+1)
      res += '\n'
      res += indent*depth + f"{right[ty]}"
    
  
  rf(data, None)
  
  
  with open(file_path, "w", encoding='utf-8') as file:
    file.write(res)

if __name__ == '__main__':  
  refine('./temp.json', max_depth=[3, 4, 0])