#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_TREE="$(cd "${SCRIPT_DIR}/.." && pwd)"
GIT_DIR="$(git -C "${WORK_TREE}" rev-parse --git-dir)"
PID_FILE="${GIT_DIR}/auto_commit_push.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  printf 'Auto sync is not running.\n'
  exit 0
fi

PID="$(cat "${PID_FILE}")"

if kill -0 "${PID}" >/dev/null 2>&1; then
  kill "${PID}"
  printf 'Stopped auto sync process %s.\n' "${PID}"
else
  printf 'Recorded PID %s is not running; cleaning up stale state.\n' "${PID}"
fi

rm -f "${PID_FILE}"
