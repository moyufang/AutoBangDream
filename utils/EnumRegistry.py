import json
from enum import Enum, IntFlag, auto

enum_registry = {}

def enum_register(cls):
  enum_registry[cls.__name__] = cls
  def __str__(self):
      return f"{self.__class__.__name__}.{self.name}"
  cls.__str__ = __str__
  return cls

class EnumEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Enum) or isinstance(obj, IntFlag): return str(obj)
    return super().default(obj)

def enum_hook(dct):
  for k, v in dct.items():
    if isinstance(v, str) and '.' in v:
      cls_name, member_name = v.split('.')
      if cls_name in enum_registry:
        dct[k] = enum_registry[cls_name][member_name]
  return dct

def enum_dumps(obj, *arg):
  return json.dumps(obj, cls = EnumEncoder, *arg)

def enum_loads(s, *arg):
  return json.loads(s, object_hook=enum_hook, *arg)

def enum_dump(obj, fp, *arg):
  json.dump(obj, fp, cls = EnumEncoder, *arg)
  
def enum_load(fp, *arg):
  return json.load(fp, object_hook=enum_hook, *arg)

if __name__ == "__main__":
  @enum_register
  class C(Enum):
    RED = auto()
    BLUE = auto()
    YELLOW = auto()
    GREEN = auto()
  config = {
    'c': C.RED,
    'fig': {
      'c':C.BLUE,
      'y':[C.YELLOW, C.GREEN]
    }
  }
  s = enum_dumps(config)
  print(s)
  v = enum_loads(s)
  print(v)