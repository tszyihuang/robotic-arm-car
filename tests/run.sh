#!/usr/bin/env bash
set -euo pipefail
CAR_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CAR_PROJECT_DIR"
python3 -B -m unittest discover -s tests -t . "$@"
