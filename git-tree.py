#!/usr/bin/env python3

import os
import subprocess
import sys

def git_tree_perfect(path="."):
    """完美版本的 Git 树形显示"""
    try:
        original_dir = os.getcwd()
        if path != ".":
            os.chdir(path)
        
        result = subprocess.run(
            ["git", "ls-files", "--exclude-standard"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            print("错误: 不是 Git 仓库")
            return
        
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        
        if not files:
            print("没有 Git 跟踪的文件")
            return
        
        print(f"Git Tree: {os.getcwd()}\n")
        
        # 使用更简单的方法：记录已显示的目录
        shown_dirs = set()
        sorted_files = sorted(files)
        
        for i, file_path in enumerate(sorted_files):
            parts = file_path.replace('\\', '/').split('/')
            
            # 显示目录结构
            for depth in range(len(parts) - 1):
                dir_path = '/'.join(parts[:depth + 1])
                if dir_path not in shown_dirs:
                    # 计算前缀
                    prefix = ""
                    for d in range(depth):
                        # 检查这个目录后面是否还有兄弟目录
                        has_sibling = any(
                            f.replace('\\', '/').split('/')[d] != parts[d] 
                            for f in sorted_files[i:] 
                            if len(f.replace('\\', '/').split('/')) > d
                        )
                        prefix += "│   " if has_sibling else "    "
                    
                    # 检查这个目录是否是其父目录的最后一个子项
                    is_last_dir = not any(
                        f.replace('\\', '/').split('/')[depth] != parts[depth] 
                        for f in sorted_files[i + 1:] 
                        if len(f.replace('\\', '/').split('/')) > depth
                    )
                    
                    connector = "└── " if is_last_dir else "├── "
                    print(f"{prefix}{connector}{parts[depth]}/")
                    shown_dirs.add(dir_path)
            
            # 显示文件
            prefix = ""
            for depth in range(len(parts) - 1):
                # 检查这个层级后面是否还有文件
                has_sibling = any(
                    f.replace('\\', '/').split('/')[depth] != parts[depth] 
                    for f in sorted_files[i + 1:] 
                    if len(f.replace('\\', '/').split('/')) > depth
                )
                prefix += "│   " if has_sibling else "    "
            
            # 检查这个文件是否是最后一个
            is_last_file = i == len(sorted_files) - 1
            connector = "└── " if is_last_file else "├── "
            print(f"{prefix}{connector}{parts[-1]}")
            
    except Exception as e:
        print(f"错误: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    git_tree_perfect(path)