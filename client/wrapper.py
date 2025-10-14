# client/wrapper.py
from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# 服务静态文件
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# API路由
@app.route('/api/<module>/<task>/start', methods=['POST'])
def start_task(module, task):
    # 任务启动逻辑
    pass
  
@app.route('/api/stop', methods=['POST'])
def stop_task():
    # 任务停止逻辑
    pass

# WebSocket支持
from flask_socketio import SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('log')
def handle_log(data):
    # 处理日志消息
    socketio.emit('log_update', data)