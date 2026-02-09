#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专用于 FedITS-Tool 的源码汇总脚本
扫描 /home/veins/fedits-tool 下所有的 .py 和 .sh 文件
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Iterable, List, Tuple

# 根据你的项目结构，定制的忽略目录
DEFAULT_IGNORE_DIRS = {
    # 标准忽略
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache", 
    "venv", ".venv", "env", ".env", "node_modules", "dist", "build",
    
    # 你项目特定的忽略
    "outputs",           # 运行结果，不用扫
    "figures",           # 图片目录
    "omnetpp_projects",  # C++ 代码通常很大，如果只需要 Python/Shell 可忽略，或者保留看里面的 Makefile
    "runs"               # 运行时生成的数据
}

def find_source_files(
    root: Path,
    exts: Tuple[str, ...] = (".py", ".sh"),
    ignore_dirs: Iterable[str] = DEFAULT_IGNORE_DIRS,
) -> List[Path]:
    ignore_dirs_set = set(ignore_dirs)
    files: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 原地修改 dirnames，防止进入忽略目录
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs_set]

        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                files.append(p)

    # 按文件路径排序，保证输出顺序一致
    files.sort(key=lambda x: str(x).lower())
    return files

def read_text_safely(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return p.read_text(errors="replace")

def choose_fence(content: str) -> str:
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
    suf = p.suffix.lower()
    if suf == ".py":
        return "python"
    if suf == ".sh":
        return "bash"
    return ""

def write_markdown(out_path: Path, root: Path, files: List[Path]) -> None:
    lines: List[str] = []
    lines.append(f"# FedITS-Tool Source Code Dump\n")
    lines.append(f"- Root: `{root.resolve()}`\n")
    lines.append(f"- Files: **{len(files)}**\n")
    lines.append(f"- Generated at: `{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n")
    lines.append("\n---\n\n")

    # 目录索引 (Index)
    if files:
        lines.append("## Index\n\n")
        for p in files:
            rel = p.relative_to(root).as_posix()
            anchor = rel.lower().replace("/", "").replace("\\", "").replace(".", "").replace(" ", "")
            lines.append(f"- [{rel}](#{anchor})\n")
        lines.append("\n---\n\n")

    # 文件内容
    for p in files:
        rel = p.relative_to(root).as_posix()
        anchor = rel.lower().replace("/", "").replace("\\", "").replace(".", "").replace(" ", "")
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")

def main() -> None:
    # 硬编码项目根目录路径
    PROJECT_ROOT = Path("/home/veins/fedits-tool")
    
    parser = argparse.ArgumentParser(description="Dump FedITS project files.")
    parser.add_argument(
        "--dir", 
        dest="target_dir", 
        default=str(PROJECT_ROOT),  # <--- 默认指向项目根目录
        help="Target directory"
    )
    parser.add_argument(
        "--out-dir", 
        dest="out_dir", 
        default=str(PROJECT_ROOT / "docs"), # <--- 默认输出到 docs 目录
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    root = Path(args.target_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    
    # 扫描
    files = find_source_files(root)
    
    # 生成文件名
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"source_dump_{ts}.md"
    
    write_markdown(out_path, root, files)
    
    print(f"-" * 40)
    print(f"[OK] 扫描路径: {root}")
    print(f"[OK] 采集文件: {len(files)} 个 (.py, .sh)")
    print(f"[OK] 输出文件: {out_path}")
    print(f"-" * 40)

if __name__ == "__main__":
    main()