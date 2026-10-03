#!/usr/bin/env bash
# ============================================
# v5.33 Render start script - phone-bridge wiring
# ============================================
# Starts the optional Tailscale userspace daemon BEFORE the bot, so the
# Render service can reach the user's phone inside the tailnet (the phone
# = exit node = a personal, clean Binance egress IP that no datacenter
# shares). Everything is non-fatal: any failure here leaves the bot
# running with direct egress (v5.33 per-request fallback also protects
# against the phone going down mid-flight).
#
# Required env (set in Render dashboard):
#   TS_AUTHKEY       - Tailscale auth key (admin console -> Keys)
#   TS_EXIT_NODE     - the phone's tailnet name or 100.x.y.z IP
#   BINANCE_PROXY_URL- socks5h://127.0.0.1:1055 (tailscaled userspace hop)
# ============================================
set -u

PORT="${PORT:-8080}"

if [ -f ./tailscaled ] && [ -n "${TS_AUTHKEY:-}" ]; then
  echo "[start] launching tailscaled (userspace networking)..."
  mkdir -p ts_state
  ./tailscaled \
    --tun=userspace-networking \
    --state=ts_state/tailscaled.state \
    --socket=ts_state/tailscaled.sock \
    --port=0 >/tmp/tailscaled.log 2>&1 &

  # v5.33.1: when the bridge is on, keep the route honest with zero
  # traffic: boot warm-up probe + scheduled re-probe while parked in a
  # 418 cooldown (the in-band lazy re-probe never fires with no callers).
  # Longer probe timeout: cold DERP-relayed handshakes need >6s.
  export BINANCE_PROXY_BG_PROBE="${BINANCE_PROXY_BG_PROBE:-true}"
  export BINANCE_PROXY_PROBE_TIMEOUT_S="${BINANCE_PROXY_PROBE_TIMEOUT_S:-15}"

  UP_ARGS="--authkey=${TS_AUTHKEY} --hostname=crypto-bot-render"
  if [ -n "${TS_EXIT_NODE:-}" ]; then
    UP_ARGS="${UP_ARGS} --exit-node=${TS_EXIT_NODE}"
  fi
  # coreutils timeout bounds the auth attempt (non-interactive authkey)
  if timeout 45 ./tailscale --socket=ts_state/tailscaled.sock up ${UP_ARGS}; then
    echo "[start] tailscale UP - phone bridge route available (exit-node: ${TS_EXIT_NODE:-none})"
  else
    echo "[start] tailscale auth/route FAILED - continuing with direct-only egress"
  fi
else
  echo "[start] TS_AUTHKEY empty or tailscaled binary missing - direct-only egress"
fi

exec uvicorn src.web.app:app --host 0.0.0.0 --port "$PORT" --workers 1
