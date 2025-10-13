// WebSocket连接处理
class LogManager {
    constructor() {
        this.ws = null;
        this.logAreas = {};
        this.init();
    }

    init() {
        this.connectWebSocket();
        
        // 重连机制
        setInterval(() => {
            if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
                this.connectWebSocket();
            }
        }, 5000);
    }

    connectWebSocket() {
        this.ws = new WebSocket(`ws://${window.location.host}/ws/logs`);
        
        this.ws.onmessage = (event) => {
            const logData = JSON.parse(event.data);
            this.addLog(logData.module, logData.message);
        };

        this.ws.onopen = () => {
            console.log('WebSocket连接已建立');
        };

        this.ws.onclose = () => {
            console.log('WebSocket连接已断开');
        };
    }

    registerLogArea(module, element) {
        this.logAreas[module] = element;
    }

    addLog(module, message) {
        const logArea = this.logAreas[module];
        if (logArea) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="text-gray-400">[${timestamp}]</span> ${message}`;
            
            logArea.appendChild(logEntry);
            logArea.scrollTop = logArea.scrollHeight;
        }
    }
}

// 初始化日志管理器
const logManager = new LogManager();

// 模块加载后注册日志区域
document.addEventListener('DOMContentLoaded', function() {
    // 监听HTMX内容切换
    document.body.addEventListener('htmx:afterSwap', function(event) {
        const module = event.detail.target.id === 'module-content' ? 
                      getCurrentModule() : null;
        
        if (module) {
            const logArea = document.getElementById(`${module}-logs`);
            if (logArea) {
                logManager.registerLogArea(module, logArea);
            }
        }
    });
});

function getCurrentModule() {
    const activeTab = document.querySelector('.tab-active');
    return activeTab ? activeTab.dataset.tab : null;
}

// 全局任务状态监控
function updateGlobalStatus() {
    // 定期检查任务状态
    setInterval(() => {
        htmx.ajax('GET', '/api/status', {
            swap: 'none'
        }).then(response => {
            // 更新全局状态显示
            const statusElement = document.getElementById('global-status');
            const taskElement = document.getElementById('current-task');
            
            if (response.active_task) {
                statusElement.classList.remove('hidden');
                taskElement.textContent = `${response.active_task} 运行中`;
            } else {
                statusElement.classList.add('hidden');
            }
        });
    }, 1000);
}

// 启动状态监控
updateGlobalStatus();