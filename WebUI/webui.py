from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import asyncio
import json
import logging

# 导入你的后端函数
from your_backend import (
    start_scriptor, stop_scriptor,
    add_song, stop_adding, train_song_recognition, stop_train_song_recognition,
    add_img, train_UI_recognition, stop_train_UI_recognition,
    fetch, workflow,
    get_avail_state, get_all_weight_title, check_song_id, get_color
)

app = FastAPI(title="Your Project WebUI")

# 挂载静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 全局状态管理
class AppState:
    def __init__(self):
        self.active_task = None  # 当前运行的任务模块
        self.log_connections: List[WebSocket] = []
        self.scriptor_config = {
            "dilation_time": 1000000,
            "correction_time": -45000,
            "mumu_port": 7555,
            "server_port": 31415,
            "bangcheater_port": 12345,
            "is_no_action": False,
            "is_caliboration": False,
            "play_one_song_id": 655,
            "is_play_one_song": False,
            "is_restart_play": True,
            "is_checking_3d": True,
            "is_repeat": True,
            "MAX_SAME_STATE": 100,
            "MAX_RE_READY": 10,
            "is_allow_save": True,
            "protected_state": ['join_wait', 'ready_done'],
            "special_state_list": ['ready'],
            "mode": "Event",
            "event": "Compete",
            "choose": "Loop",
            "level": "Expert",
            "performance": "AllPerfect",
            "weight_title": "skilled",
            "lobby": True
        }
        # 其他模块配置...

app_state = AppState()

# WebSocket连接管理
@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_state.log_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        app_state.log_connections.remove(websocket)

# 广播日志到所有连接的客户端
async def broadcast_log(message: str, module: str = "system"):
    log_data = {
        "module": module,
        "message": message,
        "timestamp": asyncio.get_event_loop().time()
    }
    disconnected = []
    for connection in app_state.log_connections:
        try:
            await connection.send_text(json.dumps(log_data))
        except:
            disconnected.append(connection)
    
    for connection in disconnected:
        app_state.log_connections.remove(connection)

# 主页 - 标签页布局
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

# Scriptor模块API
@app.post("/api/scriptor/start")
async def api_start_scriptor(config: Dict[str, Any]):
    if app_state.active_task and app_state.active_task != "scriptor":
        return {"status": "error", "message": "另一个任务正在运行"}
    
    app_state.active_task = "scriptor"
    app_state.scriptor_config.update(config)
    
    # 调用你的后端函数
    try:
        start_scriptor(config)
        await broadcast_log("Scriptor 任务已启动", "scriptor")
        return {"status": "success"}
    except Exception as e:
        app_state.active_task = None
        await broadcast_log(f"Scriptor 启动失败: {str(e)}", "scriptor")
        return {"status": "error", "message": str(e)}

@app.post("/api/scriptor/stop")
async def api_stop_scriptor():
    if app_state.active_task == "scriptor":
        stop_scriptor()
        app_state.active_task = None
        await broadcast_log("Scriptor 任务已停止", "scriptor")
        return {"status": "success"}
    return {"status": "error", "message": "Scriptor 未运行"}

# Song Recognition模块API
@app.post("/api/song_recognition/add_song")
async def api_add_song():
    if app_state.active_task and app_state.active_task != "song_recognition":
        return {"status": "error", "message": "另一个任务正在运行"}
    
    app_state.active_task = "song_recognition"
    try:
        add_song()
        await broadcast_log("开始添加歌曲", "song_recognition")
        return {"status": "success"}
    except Exception as e:
        app_state.active_task = None
        return {"status": "error", "message": str(e)}

@app.post("/api/song_recognition/stop_adding")
async def api_stop_adding_song():
    if app_state.active_task == "song_recognition":
        stop_adding()
        app_state.active_task = None
        await broadcast_log("停止添加歌曲", "song_recognition")
        return {"status": "success"}
    return {"status": "error", "message": "没有正在运行的歌曲添加任务"}

# 类似的API为其他模块...

# 工具API
@app.get("/api/utils/avail_states")
async def get_available_states():
    return get_avail_state()

@app.get("/api/utils/weight_titles")
async def get_weight_titles():
    return get_all_weight_title()

@app.post("/api/utils/check_song_id")
async def check_song_id_api(song_id: int):
    return check_song_id(song_id)

# 模块页面
@app.get("/module/{module_name}", response_class=HTMLResponse)
async def get_module(request: Request, module_name: str):
    template_map = {
        "scriptor": "scriptor.html",
        "song_recognition": "song_recognition.html",
        "ui_recognition": "ui_recognition.html",
        "fetch": "fetch.html",
        "workflow": "workflow.html"
    }
    
    if module_name in template_map:
        return templates.TemplateResponse(
            template_map[module_name], 
            {"request": request, "module": module_name}
        )
    return {"error": "模块不存在"}