#!/bin/bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_TREE="$(cd "${SCRIPT_DIR}/.." && pwd)"
GIT_DIR="$(git -C "${WORK_TREE}" rev-parse --git-dir)"

AUTO_PUSH_INTERVAL_SECONDS="${AUTO_PUSH_INTERVAL_SECONDS:-5}"
AUTO_PUSH_QUIET_SECONDS="${AUTO_PUSH_QUIET_SECONDS:-20}"
AUTO_PUSH_MESSAGE_PREFIX="${AUTO_PUSH_MESSAGE_PREFIX:-auto: sync thesis changes}"
AUTO_PUSH_BRANCH="${AUTO_PUSH_BRANCH:-$(git -C "${WORK_TREE}" branch --show-current 2>/dev/null || true)}"

if [[ -z "${AUTO_PUSH_BRANCH}" ]]; then
  AUTO_PUSH_BRANCH="main"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

status_snapshot() {
  git -C "${WORK_TREE}" status --porcelain=v1 --untracked-files=all -- .
}

push_if_ahead() {
  local counts ahead

  if ! git -C "${WORK_TREE}" rev-parse --verify "origin/${AUTO_PUSH_BRANCH}" >/dev/null 2>&1; then
    return 0
  fi

  counts="$(git -C "${WORK_TREE}" rev-list --left-right --count "origin/${AUTO_PUSH_BRANCH}...HEAD" 2>/dev/null || true)"
  ahead="$(printf '%s' "${counts}" | awk '{print $2}')"

  if [[ -n "${ahead}" && "${ahead}" != "0" ]]; then
    log "Local branch is ahead of origin/${AUTO_PUSH_BRANCH}; trying to push pending commit(s)."
    if git -C "${WORK_TREE}" push origin "${AUTO_PUSH_BRANCH}"; then
      log "Push completed."
    else
      log "Push failed; will retry on the next cycle."
    fi
  fi
}

commit_and_push() {
  local commit_message

  git -C "${WORK_TREE}" add -A -- .
  if git -C "${WORK_TREE}" diff --cached --quiet; then
    log "No staged changes after git add; skipping this round."
    return 0
  fi

  commit_message="${AUTO_PUSH_MESSAGE_PREFIX} ($(date '+%Y-%m-%d %H:%M:%S'))"
  log "Creating commit: ${commit_message}"

  if ! git -C "${WORK_TREE}" commit -m "${commit_message}"; then
    log "Commit failed; keeping changes in the working tree."
    return 1
  fi

  if git -C "${WORK_TREE}" push origin "${AUTO_PUSH_BRANCH}"; then
    log "Push completed."
  else
    log "Push failed after commit; the branch is ahead locally and will be retried."
    return 1
  fi
}

cleanup() {
  local pid_file
  pid_file="${GIT_DIR}/auto_commit_push.pid"
  rm -f "${pid_file}"
  log "Auto sync stopped."
}

trap cleanup EXIT INT TERM

log "Watching ${WORK_TREE} on branch ${AUTO_PUSH_BRANCH}."
log "Polling every ${AUTO_PUSH_INTERVAL_SECONDS}s with a ${AUTO_PUSH_QUIET_SECONDS}s quiet window."

last_digest=""
last_change_at=0

while true; do
  current_status="$(status_snapshot)"

  if [[ -n "${current_status}" ]]; then
    current_digest="$(printf '%s' "${current_status}" | shasum -a 256 | awk '{print $1}')"

    if [[ "${current_digest}" != "${last_digest}" ]]; then
      last_digest="${current_digest}"
      last_change_at="$(date +%s)"
      log "Detected working tree changes; waiting for the quiet window to pass."
    else
      now="$(date +%s)"
      if (( now - last_change_at >= AUTO_PUSH_QUIET_SECONDS )); then
        commit_and_push || true
        last_digest=""
        last_change_at=0
      fi
    fi
  else
    last_digest=""
    last_change_at=0
    push_if_ahead
  fi

  sleep "${AUTO_PUSH_INTERVAL_SECONDS}"
done
