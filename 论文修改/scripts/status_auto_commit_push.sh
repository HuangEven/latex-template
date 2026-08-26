#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_TREE="$(cd "${SCRIPT_DIR}/.." && pwd)"
GIT_DIR="$(git -C "${WORK_TREE}" rev-parse --git-dir)"
PID_FILE="${GIT_DIR}/auto_commit_push.pid"
LOG_FILE="${GIT_DIR}/auto_commit_push.log"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" >/dev/null 2>&1; then
    printf 'Auto sync is running with PID %s.\n' "${PID}"
  else
    printf 'Auto sync is not running, but a stale PID file exists.\n'
  fi
else
  printf 'Auto sync is not running.\n'
fi

printf 'Repository: %s\n' "$(git -C "${WORK_TREE}" rev-parse --show-toplevel)"
printf 'Working tree: %s\n' "${WORK_TREE}"
printf 'Branch: %s\n' "$(git -C "${WORK_TREE}" branch --show-current)"
printf 'Log file: %s\n' "${LOG_FILE}"
