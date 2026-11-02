# 在 Windows PowerShell 中，于项目根目录（E:\硕士毕业论文最新版）运行：
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# 前提：已安装 Python 3.11+ 与 Docker Desktop，且 Docker Desktop 正在运行。
$ErrorActionPreference = "Stop"
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
docker pull mysql:8.0.36
Write-Host "环境就绪。运行测试：.\.venv\Scripts\python -m pytest -q tests"
Write-Host "含数据库的测试：`$env:PRISM_TEST_MYSQL='1'; .\.venv\Scripts\python -m pytest -q tests"
