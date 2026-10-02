#!/usr/bin/env bash
# Start the DSH policy gateway proxy (port 8766) and a Cloudflare quick tunnel for the fusion run,
# following the P0 start_services.py contract: private route dir, public origin, 401/404 auth probe.
# usage: start_dsh_services.sh <run_dir>
set -euo pipefail
RUN="${1:?run dir}"
PREFIX=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
SCRIPTS=/workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts
source "$PREFIX/activate.sh" "$PREFIX"
source "$SCRIPTS/runtime.env"
source /workspace/xdan-verl-fusion/ops/fusion-runtime.env
TUNNEL_BIN=/workspace/train-p0-dsh-integration/tools/cloudflared
mkdir -p "$RUN"
umask 077; mkdir -p /root/mimo-private/dsh-routes; umask 022
for f in proxy.pid tunnel.pid public-origin.txt; do test ! -e "$RUN/$f"; done

cd "$SOURCE"
nohup python -m recipes.code.dsh_gateway_proxy --route-dir /root/mimo-private/dsh-routes --port 8766 \
  > "$RUN/proxy.log" 2>&1 < /dev/null &
echo $! > "$RUN/proxy.pid"
nohup "$TUNNEL_BIN" tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8766 \
  > "$RUN/tunnel.log" 2>&1 < /dev/null &
echo $! > "$RUN/tunnel.pid"

origin=""
for _ in $(seq 1 60); do
  origin=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$RUN/tunnel.log" | head -1 || true)
  [ -n "$origin" ] && break
  sleep 2
done
test -n "$origin" || { echo "no public origin"; exit 1; }
echo "$origin" > "$RUN/public-origin.txt"

probe() {  # url expected
  for _ in $(seq 1 15); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' --max-time 10 "$1" || true)
    [ "$code" = "$2" ] && { echo "ok $2 $1"; return 0; }
    sleep 3
  done
  echo "FAIL expected $2 got $code for $1"; return 1
}
for base in http://127.0.0.1:8766 "$origin"; do
  probe "$base/sessions/fusion-unauthorized-probe/v1/chat/completions" 401
  probe "$base/fusion-unknown-path" 404
done
echo "services ready: $origin"
