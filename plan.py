#!/usr/bin/env python3
"""任务表入口：默认执行主线，使用 --分段名 选择任务表中的其他分段。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src/car_nodes'))
from car_nodes.planner.client import main


if __name__ == '__main__':
    raise SystemExit(main())
