#!/usr/bin/env python3

import os
import subprocess
import sys

def git_tree_simple(path="."):
    """简化版本的 Git 树形显示"""
    try:
        original_dir = os.getcwd()
        if path != ".":
            os.chdir(path)
        
        # 获取 Git 文件列表
        result = subprocess.run(
            ["git", "ls-files", "--exclude-standard"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            print("错误: Git 命令执行失败")
            return
        
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        
        if not files:
            print("没有 Git 跟踪的文件")
            return
        
        print(f"Git Tree: {os.getcwd()}\n")
        
        # 使用集合记录已显示的目录
        shown_dirs = set()
        
        for file_path in sorted(files):
            parts = file_path.replace('\\', '/').split('/')
            
            # 显示目录结构
            for i in range(len(parts) - 1):
                dir_path = '/'.join(parts[:i+1])
                if dir_path not in shown_dirs:
                    indent = "    " * i
                    connector = "└── " if i == 0 else "├── "
                    print(f"{indent}{connector}{parts[i]}/")
                    shown_dirs.add(dir_path)
            
            # 显示文件
            indent = "    " * (len(parts) - 1)
            connector = "└── " if len(parts) == 1 else "├── "
            print(f"{indent}{connector}{parts[-1]}")
            
    except Exception as e:
        print(f"错误: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    git_tree_simple(path)