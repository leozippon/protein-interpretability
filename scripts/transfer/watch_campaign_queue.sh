#!/usr/bin/env bash
set -euo pipefail

# Poll a detached in-pod campaign status file until every cell is terminal.
# H200_POD is shell-local; this script does not write pod names.

INTERVAL=600
CAMPAIGN="campaign_r240_extra_context"
STATUS=""
H200="${H200:-$HOME/hangzhou-compute/h200}"

usage() {
  echo "usage: H200_POD=<pod> $0 [--interval 600] [--campaign campaign_r240_extra_context] [--status <gpfs-status.tsv>]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --status) STATUS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

[[ -n "${H200_POD:-}" ]] || { echo "H200_POD is not set" >&2; exit 2; }
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "interval must be a positive integer" >&2; exit 2; }
if [[ -z "$STATUS" ]]; then
  STATUS="/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/logs/external_baseline/${CAMPAIGN}.status.tsv"
fi

remote_head() {
  "$H200" exec -- head -n 16 "$STATUS"
}

parse_tally() {
  local text="$1"
  python3 -c '
import re, sys
text = sys.stdin.read()
upd = re.search(r"^# updated_utc\t(.+)$", text, re.M)
slot = re.search(r"^# slot\t(.+)$", text, re.M)
tally = re.search(r"^# tally\t(.+)$", text, re.M)
fail = re.search(r"^# FAILURES\t(.+)$", text, re.M)
print("updated", (upd.group(1).strip() if upd else "?"))
print("slot", (slot.group(1).strip() if slot else "?"))
print("tally", (tally.group(1).strip() if tally else "?"))
print("failures", (fail.group(1).strip() if fail else "?"))
fields = dict(
    part.split("=", 1)
    for part in (tally.group(1).split() if tally else [])
    if "=" in part
)
pending = int(fields.get("pending", "1"))
running = int(fields.get("running", "1"))
ok = int(fields.get("exited-ok", "0"))
bad = int(fields.get("exited-nonzero", "0"))
print("pending", pending)
print("running", running)
print("ok", ok)
print("nonzero", bad)
sys.exit(0 if pending == 0 and running == 0 else 1)
' <<<"$text"
}

echo "watching ${STATUS} every ${INTERVAL}s"
while true; do
  head_txt="$(remote_head)"
  echo "---- $(date -u +%Y-%m-%dT%H:%M:%SZ) ----"
  echo "$head_txt" | sed -n '1,12p'
  if parse_tally "$head_txt"; then
    echo "CAMPAIGN_TERMINAL ${CAMPAIGN}"
    exit 0
  fi
  sleep "$INTERVAL"
done
