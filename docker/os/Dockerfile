# 操作系统环境的镜像（与 LifelongAgentBench 官方 local-os/default 的做法相同：Ubuntu 加常用工具）。
# 固定发行版版本；正式实验前执行 docker image inspect 记录镜像 ID。
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip git vim curl wget unzip zip tree sudo acl cron \
    && rm -rf /var/lib/apt/lists/*
