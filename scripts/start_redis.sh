#!/usr/bin/env bash
# =============================================================================
# Cook Agent - Redis 本地一键启动脚本
#
# 用途: 安装（如缺失）并启动本地 Redis，供 StorageBackend(redis) 使用。
#
# 行为:
#   1. 若 ~/opt/redis/redis-server 不存在，从官方下载源码编译安装
#      （macOS 无需 brew/docker；Linux 同样适用）
#   2. 若端口未监听，后台启动 redis-server（--daemonize）
#   3. 打印连接状态
#
# 用法:
#   ./scripts/start_redis.sh            # 默认端口 6379
#   REDIS_PORT=6380 ./scripts/start_redis.sh
#   REDIS_HOME=/custom/path ./scripts/start_redis.sh
#
# 停止:
#   $HOME/opt/redis/redis-cli -p 6379 shutdown
# =============================================================================
set -euo pipefail

REDIS_VERSION="${REDIS_VERSION:-7.2.5}"
REDIS_HOME="${REDIS_HOME:-$HOME/opt/redis}"
REDIS_SERVER="$REDIS_HOME/redis-server"
REDIS_CLI="$REDIS_HOME/redis-cli"
PORT="${REDIS_PORT:-6379}"

# ---------------------------------------------------------------------------
# 1. 安装（若缺失）
# ---------------------------------------------------------------------------
if [ ! -x "$REDIS_SERVER" ]; then
  echo "==> Redis 未安装，正在下载并编译 Redis ${REDIS_VERSION} ..."
  echo "    安装目录: ${REDIS_HOME}"
  mkdir -p "$REDIS_HOME"
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR"' EXIT

  curl -sL -o "$TMPDIR/redis.tar.gz" \
    "https://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz"
  tar -xzf "$TMPDIR/redis.tar.gz" -C "$TMPDIR"

  echo "==> 编译中（约 1 分钟，无需管理员权限）..."
  (cd "$TMPDIR/redis-${REDIS_VERSION}" && make -j4 BUILD_TLS=no >/dev/null 2>&1)

  cp "$TMPDIR/redis-${REDIS_VERSION}/src/redis-server" "$REDIS_HOME/"
  cp "$TMPDIR/redis-${REDIS_VERSION}/src/redis-cli" "$REDIS_HOME/"
  echo "==> Redis ${REDIS_VERSION} 安装完成: ${REDIS_HOME}"
fi

# ---------------------------------------------------------------------------
# 2. 启动（若未监听）
# ---------------------------------------------------------------------------
if nc -z localhost "$PORT" 2>/dev/null; then
  echo "==> Redis 已在运行: 端口 ${PORT}"
else
  echo "==> 启动 Redis: 端口 ${PORT}"
  "$REDIS_SERVER" --port "$PORT" --daemonize yes --save "" --appendonly no
  sleep 1
  if ! nc -z localhost "$PORT" 2>/dev/null; then
    echo "ERROR: Redis 启动失败" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 3. 状态确认
# ---------------------------------------------------------------------------
"$REDIS_CLI" -p "$PORT" ping
"$REDIS_CLI" -p "$PORT" info server | grep -E "redis_version" | head -1
echo ""
echo "✅ Redis 就绪: 127.0.0.1:${PORT}"
echo "   停止命令: ${REDIS_HOME}/redis-cli -p ${PORT} shutdown"
