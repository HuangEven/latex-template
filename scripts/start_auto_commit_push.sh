#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_TREE="$(cd "${SCRIPT_DIR}/.." && pwd)"
GIT_DIR="$(git -C "${WORK_TREE}" rev-parse --git-dir)"
PID_FILE="${GIT_DIR}/auto_commit_push.pid"
LOG_FILE="${GIT_DIR}/auto_commit_push.log"

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if kill -0 "${EXISTING_PID}" >/dev/null 2>&1; then
    printf 'Auto sync is already running with PID %s.\n' "${EXISTING_PID}"
    printf 'Log file: %s\n' "${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

nohup "${SCRIPT_DIR}/auto_commit_push.sh" </dev/null >> "${LOG_FILE}" 2>&1 &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
printf '%s\n' "${NEW_PID}" > "${PID_FILE}"

printf 'Started auto sync with PID %s.\n' "${NEW_PID}"
printf 'Log file: %s\n' "${LOG_FILE}"
