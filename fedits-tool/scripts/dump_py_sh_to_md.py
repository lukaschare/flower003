#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect all .py and .sh files under a directory and dump their contents into one Markdown file.
Output filename uses timestamp, e.g., 2026-02-08_19-14-30.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "dist",
    "build",
    "outputs",   # 你的项目里 outputs/ 通常不想扫
}


def find_source_files(
    root: Path,
    exts: Tuple[str, ...] = (".py", ".sh"),
    ignore_dirs: Iterable[str] = DEFAULT_IGNORE_DIRS,
) -> List[Path]:
    ignore_dirs_set = set(ignore_dirs)
    files: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 原地修改 dirnames，避免进入忽略目录
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs_set]

        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                files.append(p)

    files.sort(key=lambda x: str(x).lower())
    return files


def read_text_safely(p: Path) -> str:
    """
    Try reading file as UTF-8; fallback to system default if needed.
    Always returns a string (with replacement on decode errors).
    """
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return p.read_text(errors="replace")


def choose_fence(content: str) -> str:
    """
    Choose a backtick fence that won't conflict with content.
    If content contains ``` then use ```` etc.
    """
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


def timestamp_name(style: str) -> str:
    now = dt.datetime.now()
    if style == "dash":
        # 2026-02-08_19-14-30 (跨平台安全)
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    if style == "compact":
        # 20260208_191430
        return now.strftime("%Y%m%d_%H%M%S")
    if style == "cn":
        # 2026年02月08日19时14分30秒（Windows也基本可用，但更“花”）
        return now.strftime("%Y年%m月%d日%H时%M分%S秒")
    # default
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def write_markdown(out_path: Path, root: Path, files: List[Path]) -> None:
    lines: List[str] = []
    lines.append(f"# Source dump\n")
    lines.append(f"- Root: `{root.resolve()}`\n")
    lines.append(f"- Files: **{len(files)}**\n")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(sep=' ', timespec='seconds')}`\n")
    lines.append("\n---\n\n")

    # Index
    if files:
        lines.append("## Index\n\n")
        for p in files:
            rel = p.relative_to(root).as_posix()
            anchor = rel.lower().replace("/", "").replace("\\", "").replace(".", "").replace(" ", "")
            lines.append(f"- [{rel}](#{anchor})\n")
        lines.append("\n---\n\n")

    for p in files:
        rel = p.relative_to(root).as_posix()
        anchor = rel.lower().replace("/", "").replace("\\", "").replace(".", "").replace(" ", "")
        content = read_text_safely(p)
        fence = choose_fence(content)
        lang = lang_for_file(p)

        lines.append(f"## {rel}\n")
        lines.append(f"<a id=\"{anchor}\"></a>\n\n")
        lines.append(f"- Path: `{p.resolve()}`\n")
        lines.append(f"- Size: **{p.stat().st_size}** bytes\n\n")
        lines.append(f"{fence}{lang}\n")
        lines.append(content)
        if not content.endswith("\n"):
            lines.append("\n")
        lines.append(f"{fence}\n\n")
        lines.append("---\n\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump .py/.sh files under a directory into one timestamped Markdown file."
    )
    parser.add_argument(
        "--dir",
        dest="target_dir",
        default="docs/figures",
        help="Target directory to scan (default: docs/figures)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default="docs/figures",
        help="Output directory for the Markdown file (default: docs/figures)",
    )
    parser.add_argument(
        "--name-style",
        dest="name_style",
        choices=["dash", "compact", "cn"],
        default="dash",
        help="Timestamp style for filename (default: dash)",
    )
    parser.add_argument(
        "--ext",
        dest="exts",
        default=".py,.sh",
        help="Comma-separated extensions to include (default: .py,.sh)",
    )
    parser.add_argument(
        "--ignore-dirs",
        dest="ignore_dirs",
        default=",".join(sorted(DEFAULT_IGNORE_DIRS)),
        help="Comma-separated dir names to ignore",
    )

    args = parser.parse_args()

    root = Path(args.target_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[ERROR] target dir not found or not a dir: {root}")

    out_dir = Path(args.out_dir).resolve()
    exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())
    ignore_dirs = [d.strip() for d in args.ignore_dirs.split(",") if d.strip()]

    files = find_source_files(root=root, exts=exts, ignore_dirs=ignore_dirs)
    ts = timestamp_name(args.name_style)
    out_path = out_dir / f"{ts}.md"

    write_markdown(out_path=out_path, root=root, files=files)

    print(f"[OK] Collected {len(files)} files")
    print(f"[OK] Output: {out_path}")


if __name__ == "__main__":
    main()
