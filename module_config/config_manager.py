import json
from configuration import *
from utils.EnumRegistry import *
from typing import Dict, Any

from abc import ABC, abstractmethod

class Config(ABC):
  @abstractmethod
  def __init__(self, config_path:str):
     self.config_path = config_path
  @abstractmethod
  def load(self):
    with open(self.config_path, 'r', encoding='utf-8') as f:
      self.cfg = enum_load(f)
  @abstractmethod
  def save(self):
    with open(self.config_path, 'w', encoding='utf-8') as f:
      enum_dump(self.cfg, f)
  def update(self, new_cfg:dict):
    for k,v in new_cfg.items():
      if hasattr(self, k):
        self[k] = v
    self.save()

config_registry = {}

def config_register(module_name:str, config_path:str):
  def _config_register(cls:Config):
    config_registry[module_name] = cls(config_path)
    return cls
  return _config_register

