#!/usr/bin/env bash
# Linux / macOS / WSL：在项目根目录运行 bash scripts/setup_unix.sh
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
docker pull mysql:8.0.36
echo "环境就绪。运行测试：PRISM_TEST_MYSQL=1 .venv/bin/python -m pytest -q tests"
