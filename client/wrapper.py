# client/wrapper.py
import os
from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

# 服务前端静态文件
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# API 路由 - Scriptor 模块
@app.route('/api/scriptor/available-states', methods=['GET'])
def get_available_states():
    return jsonify({
        'success': True,
        'data': ['join_wait', 'ready_done', 'ready', 'playing', 'finished', 'result']
    })

@app.route('/api/scriptor/start_scriptor/start', methods=['POST'])
def start_scriptor():
    config = request.get_json()
    # 调用实际的后端逻辑
    thread = threading.Thread(target=run_scriptor_task, args=(config,))
    thread.start()
    return jsonify({'success': True, 'data': {'taskId': '123', 'status': 'running'}})

def run_scriptor_task(config):
    """运行实际的脚本任务"""
    # 模拟任务执行
    for i in range(10):
        time.sleep(1)
        # 通过 WebSocket 发送日志
        socketio.emit('log', {
            'type': 'log',
            'module': 'scriptor',
            'task': 'start_scriptor',
            'data': {
                'level': 'info',
                'message': f'任务进度: {i+1}/10'
            },
            'timestamp': time.time()
        })
    socketio.emit('log', {
        'type': 'log', 
        'module': 'scriptor',
        'task': 'start_scriptor',
        'data': {
            'level': 'info', 
            'message': '任务完成'
        },
        'timestamp': time.time()
    })

# WebSocket 事件
@socketio.on('connect')
def handle_connect():
    print('客户端连接成功')
    emit('connected', {'message': 'Connected to backend'})

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端断开连接')

if __name__ == '__main__':
    # 确保静态文件目录存在
    os.makedirs('static', exist_ok=True)
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)