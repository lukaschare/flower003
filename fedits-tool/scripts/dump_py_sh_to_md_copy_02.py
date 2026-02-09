#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FedITS-Tool 全项目源码汇总脚本 (最终净化版)
功能：
1. 扫描 /home/veins/fedits-tool 下的文件
2. 包含后缀: .py, .sh, .yaml, .yml, .md, .txt, .toml, .ini
3. 包含特定文件: Dockerfile, Makefile, CITATION.cff
4. 自动忽略: outputs, runs, figures, .git, venv 等
5. 【新增】自动屏蔽采集脚本本身及其他指定文件
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Iterable, List, Tuple

# =================配置区域=================

# 1. 忽略的文件夹
DEFAULT_IGNORE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache", 
    "venv", ".venv", "env", ".env", "node_modules", "dist", "build",
    "outputs",           # 运行结果
    "figures",           # 图片
    "omnetpp_projects",  # C++工程太大，通常略过
    "runs"               # 运行时生成的数据
}

# 2. 【新增】必须忽略的具体文件名 (在这里添加你不想扫的文件)
IGNORE_FILENAMES = {
    "dump_py_sh_to_md.py",
    "dump_py_sh_to_md_copy.py",
    "dump_py_sh_to_md_copy_02.py",
    "dump_clean.py",       # 忽略脚本自己
    ".DS_Store"            # Mac系统垃圾文件
}

# 3. 必须包含的特定文件名 (白名单，即使没有后缀)
INCLUDE_FILENAMES = {
    "Dockerfile", 
    "Makefile", 
    "requirements.txt", 
    "CITATION.cff",
    "docker-compose.yml"
}

# 4. 必须包含的后缀名
INCLUDE_EXTS = {
    ".py", ".sh", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini", ".conf"
}

# =================功能函数=================

def find_source_files(
    root: Path,
    ignore_dirs: Iterable[str] = DEFAULT_IGNORE_DIRS,
    ignore_files: Iterable[str] = IGNORE_FILENAMES,
) -> List[Path]:
    ignore_dirs_set = set(ignore_dirs)
    ignore_files_set = set(ignore_files)
    files: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 原地修改 dirnames，防止进入忽略目录
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs_set]

        for name in filenames:
            # 【新增逻辑】如果在忽略文件名单里，直接跳过
            if name in ignore_files_set:
                continue

            p = Path(dirpath) / name
            
            # 判断逻辑：后缀匹配 OR 文件名在白名单 OR 是Dockerfile变体
            is_valid_ext = p.suffix.lower() in INCLUDE_EXTS
            is_valid_name = (name in INCLUDE_FILENAMES) or name.startswith("Dockerfile")
            
            if is_valid_ext or is_valid_name:
                files.append(p)

    # 排序：保证每次生成顺序一致
    files.sort(key=lambda x: str(x).lower())
    return files


def read_text_safely(p: Path) -> str:
    """尝试以 UTF-8 读取，失败则替换字符"""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return f"[Error reading file: {p.name}]"


def choose_fence(content: str) -> str:
    """防止 Markdown 代码块冲突，动态选择反引号长度"""
    max_run = 0
    run = 0
    for ch in content:
        if ch == "`":
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    fence_len = max(3, max_run + 1)
    return "`" * fence_len


def lang_for_file(p: Path) -> str:
    """根据文件特征返回 Markdown 语法高亮标签"""
    suf = p.suffix.lower()
    name = p.name.lower()
    
    if suf == ".py": return "python"
    if suf == ".sh": return "bash"
    if suf in (".yaml", ".yml"): return "yaml"
    if suf == ".md": return "markdown"
    if suf == ".toml": return "toml"
    if suf == ".ini": return "ini"
    if "dockerfile" in name: return "dockerfile"
    if "makefile" in name: return "makefile"
    return ""


def write_markdown(out_path: Path, root: Path, files: List[Path]) -> None:
    lines: List[str] = []
    lines.append(f"# FedITS-Tool Full Project Dump\n")
    lines.append(f"- Root: `{root.resolve()}`\n")
    lines.append(f"- Files: **{len(files)}**\n")
    lines.append(f"- Generated at: `{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n")
    lines.append("\n---\n\n")

    # 1. 目录索引
    if files:
        lines.append("## Index\n\n")
        for p in files:
            rel = p.relative_to(root).as_posix()
            # 生成安全的锚点链接
            anchor = rel.lower().replace("/", "").replace("\\", "").replace(".", "").replace(" ", "").replace("-", "").replace("_", "")
            lines.append(f"- [{rel}](#{anchor})\n")
        lines.append("\n---\n\n")

    # 2. 文件内容
    for p in files:
        rel = p.relative_to(root).as_posix()
        anchor = rel.lower().replace("/", "").replace("\\", "").replace(".", "").replace(" ", "").replace("-", "").replace("_", "")
        
        content = read_text_safely(p)
        fence = choose_fence(content)
        lang = lang_for_file(p)

        lines.append(f"## {rel}\n")
        lines.append(f"<a id=\"{anchor}\"></a>\n\n")
        lines.append(f"- Path: `{p}`\n")
        lines.append(f"{fence}{lang}\n")
        lines.append(content)
        if not content.endswith("\n"):
            lines.append("\n")
        lines.append(f"{fence}\n\n")
        lines.append("---\n\n")

    # 写入文件
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    # 硬编码项目根目录路径
    PROJECT_ROOT = Path("/home/veins/fedits-tool")
    
    parser = argparse.ArgumentParser(description="Dump FedITS project files.")
    parser.add_argument(
        "--dir", 
        dest="target_dir", 
        default=str(PROJECT_ROOT), 
        help="Target directory"
    )
    parser.add_argument(
        "--out-dir", 
        dest="out_dir", 
        default=str(PROJECT_ROOT / "docs"), 
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    root = Path(args.target_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    
    # 执行扫描
    if not root.exists():
        print(f"[ERROR] 目录不存在: {root}")
        return

    files = find_source_files(root)
    
    # 生成带时间戳的文件名
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"full_dump_clean_{ts}.md"
    
    # 写入
    write_markdown(out_path, root, files)
    
    print(f"-" * 40)
    print(f"[OK] 扫描路径: {root}")
    print(f"[OK] 采集文件: {len(files)} 个 (已过滤工具脚本)")
    print(f"[OK] 输出文件: {out_path}")
    print(f"-" * 40)


if __name__ == "__main__":
    main()