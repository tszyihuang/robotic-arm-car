#!/usr/bin/env python3
"""执行主线；plan 节点按扫码和当前画面选择子任务；--分段名 可单独调试。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src/car_nodes'))
from car_nodes.planner.client import main


if __name__ == '__main__':
    raise SystemExit(main())
