#!/usr/bin/env bash
# ===========================================================================
# docker-entrypoint.sh — SEC-GenerativeSearch API container init
# ===========================================================================
# Runs as root for exactly one job: make the (possibly host-mounted, and so
# root-owned) data + model-cache volumes writable by the unprivileged
# ``appuser``, then drop privileges via ``gosu`` and exec the server. The
# final, long-lived process is NEVER root.
#
# ``exec`` is used so the server replaces this shell as PID 1 and receives
# SIGTERM directly for a graceful shutdown (TaskManager cancels in-flight
# ingests on shutdown).
set -euo pipefail

APP_USER="${APP_USER:-appuser}"
APP_DATA_DIR="${APP_DATA_DIR:-/app/data}"
HF_HOME="${HF_HOME:-/app/data/hf}"

# Privileged init path. If the orchestrator already pinned a non-root user
# (e.g. Kubernetes ``runAsUser``), there is nothing to chown and no privilege
# to drop — skip straight to exec.
if [ "$(id -u)" = "0" ]; then
    mkdir -p "${APP_DATA_DIR}" "${HF_HOME}"
    # Re-own the mount points so a freshly attached, root-owned volume is
    # writable by the service account. Bounded by the volume's file count;
    # cheap on first boot.
    chown -R "${APP_USER}:${APP_USER}" "${APP_DATA_DIR}" "${HF_HOME}"
    exec gosu "${APP_USER}" "$@"
fi

exec "$@"
