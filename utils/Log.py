import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """设置统一的日志配置
    
    Args:
        name: 日志器名称，通常是 __name__
        level: 日志级别，默认INFO
        log_file: 日志文件路径，如不提供则只输出到控制台
        format_string: 自定义日志格式
    
    Returns:
        配置好的日志器实例
    """
    if format_string is None:
        format_string = (
            '[%(levelname)-5s @ %(filename)s:%(lineno)d] %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果提供了文件路径）
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    # 防止日志传递到根日志器
    logger.propagate = False
    
    return logger

logger = setup_logging(
  name=__name__,
  level=logging.DEBUG,
)
LogE = logger.error
LogD = logger.debug
LogI = logger.info

status_logger = setup_logging(
  name=__name__,
  level=logging.DEBUG,
  format_string='[STAUS @ %(filename)s:%(lineno)d] %(message)s'
)
LogS = logger.info