#============ Backend -> Frontend ============#

# 状态更新
{
  "type": "status_update",
  "module": "scriptor", 
  "data": {
    "running": True,
    "lock_holder": "scriptor",
    "config": {  }
  }
}

# 日志输出
{
  "type": "log",
  "module": "song_recognition",
  "data": {
    "level": "info", # info, warning, error
    "message": "开始训练模型...",
    "requires_input": False, # 是否需要用户输入
    "input_type": "buttons", # buttons, text, both
    "available_commands": ["Yes", "No", "Drop", "Stop"] # 可用的按钮
  }
}

# 锁状态
{
  "type": "lock_status", 
  "data": {
    "player": {"available": False, "holder": "scriptor"},
    "ui": {"available": True, "holder": None}
  }
}

# 配置响应
{
  "type": "config_response",
  "module": "scriptor",
  "data": {
    "success": True,
    "config": {  }
  }
}

# 图片数据
{
  "type": "image_data",
  "module": "workflow", 
  "data": {
    "image_base64": "base64编码的图片数据",
    "width": 1280,
    "height": 720
  }
}

#============ Frontend -> Backend ============#

# 启动/停止模块
{
  "type": "control",
  "module": "scriptor",
  "data": {
    "action": "start", # start, stop
    "config": {  }
  }
}

# 更新配置
{
  "type": "update_config",
  "module": "scriptor", 
  "data": {
    "dilation_time": 1000000,
    "is_no_action": False,
    "mode": "Event"
  }
}

# 更新配置特殊配置
{
  "type": "update_config",
  "module": "scriptor", 
  "data": {
    "add_weights":{
      "my_title": [0.97, 0.02, 0.00, 0.00, 0.01]
    },
    "del_weights":{
      "my_title": [0,0,0,0,0]
    }
  },
  "note": "update_weights_map"
}

# 用户交互响应
{
  "type": "user_input",
  "module": "song_recognition",
  "data": {
    "command": "Yes", # 或 "No", "Drop", "Stop"
    "input_data": "123" # 当command为"No"时的song_id
  }
}

# 获取配置
{
  "type": "get_config", 
  "module": "scriptor"
}

# 获取状态
{
  "type": "get_status"
}