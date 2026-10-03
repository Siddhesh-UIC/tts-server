#!/bin/bash
# GPU VM status: GPU, CPU, RAM, disk, model services and vLLM load.
# Usage: status.sh          one snapshot
#        status.sh -w [N]   refresh every N seconds (default 5), Ctrl+C to stop
#        status.sh -d       also show disk usage per /workspace folder (slower)

WS=${WS:-/workspace}
SERVICES="llm:8000 stt:8001 tts:8002"
WATCH=0; INTERVAL=5; DETAIL=0
while [ $# -gt 0 ]; do
  case "$1" in
    -w) WATCH=1; [[ "${2:-}" =~ ^[0-9]+$ ]] && { INTERVAL=$2; shift; } ;;
    -d) DETAIL=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
  esac; shift
done

B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; N=$'\e[0m'
hdr() { printf "\n${B}== %s ==${N}\n" "$1"; }
pct_color() { local p=${1%.*}; [ -z "$p" ] && p=0
  if [ "$p" -ge 90 ]; then printf "%s" "$R"; elif [ "$p" -ge 75 ]; then printf "%s" "$Y"; else printf "%s" "$G"; fi; }
bar() { local p=${1%.*}; [ -z "$p" ] && p=0; [ "$p" -gt 100 ] && p=100
  local f=$((p/5)); printf "%s[" "$(pct_color "$p")"; printf "%${f}s" "" | tr ' ' '#'
  printf "%$((20-f))s" "" | tr ' ' '.'; printf "] %3s%%%s" "$p" "$N"; }

cpu_pct() {  # CPU busy % over 1 second from /proc/stat
  read -r _ a b c d e f g h _ < /proc/stat 2>/dev/null || { echo "?"; return; }
  local t1=$((a+b+c+d+e+f+g+h)) i1=$((d+e)); sleep 1
  read -r _ a b c d e f g h _ < /proc/stat
  local t2=$((a+b+c+d+e+f+g+h)) i2=$((d+e)) dt
  dt=$((t2-t1)); [ "$dt" -le 0 ] && { echo 0; return; }
  echo $(( (100*(dt-(i2-i1)))/dt ))
}

metric() {  # metric <port> <name1> [name2...]: sum a Prometheus metric from vLLM /metrics
  local port=$1; shift; local data; data=$(curl -s -m 2 "localhost:$port/metrics") || return
  for n in "$@"; do
    local v; v=$(printf "%s\n" "$data" | awk -v n="$n" '$1==n || index($1, n"{")==1 {s+=$2; f=1} END{if(f) print s}')
    [ -n "$v" ] && { echo "$v"; return; }
  done
}

snapshot() {
  printf "${B}GPU VM status${N}  %s  host %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$(hostname)"

  hdr "GPU"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit \
      --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx name used total util temp pw pl; do
      used=${used// /}; total=${total// /}; util=${util// /}
      local_pct=$(( used*100/total ))
      printf "GPU%s %s\n" "$idx" "$(echo $name)"
      printf "  VRAM   %s  %s / %s GiB used, %s GiB free\n" "$(bar $local_pct)" \
        "$(awk "BEGIN{printf \"%.1f\",$used/1024}")" "$(awk "BEGIN{printf \"%.1f\",$total/1024}")" \
        "$(awk "BEGIN{printf \"%.1f\",($total-$used)/1024}")"
      printf "  Util   %s\n" "$(bar $util)"
      printf "  Temp   %s C    Power %s / %s W\n" "$(echo $temp)" "$(echo ${pw%.*})" "$(echo ${pl%.*})"
    done
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)
    # In this sandbox every GPU process is reported as PID 1 (env-injector) with the total VRAM; hide that
    if [ -n "$apps" ] && printf "%s\n" "$apps" | grep -qv '^1,'; then
      echo "  Processes (pid, VRAM, name):"; printf "%s\n" "$apps" | sed 's/^/    /'
    else echo "  (per-process VRAM not visible inside this container)"; fi
  else echo "nvidia-smi not found"; fi

  hdr "CPU / RAM"
  local ncpu cpu la
  ncpu=$(nproc); cpu=$(cpu_pct); read -r l1 l5 l15 _ < /proc/loadavg
  printf "CPU    %s  (%s vCPU)   load avg %s %s %s\n" "$(bar $cpu)" "$ncpu" "$l1" "$l5" "$l15"
  read -r mt mu ma < <(free -b | awk '/^Mem:/{print $2,$3,$7}')
  printf "RAM    %s  %s / %s GiB used, %s GiB available\n" "$(bar $(( mu*100/mt )))" \
    "$(awk "BEGIN{printf \"%.1f\",$mu/2^30}")" "$(awk "BEGIN{printf \"%.1f\",$mt/2^30}")" "$(awk "BEGIN{printf \"%.1f\",$ma/2^30}")"

  hdr "Disk"
  df -P -B1 "$WS" /dev/shm 2>/dev/null | awk 'NR>1{print $6,$2,$3,$4}' | while read -r mnt size used avail; do
    [ "$size" -gt 0 ] 2>/dev/null || continue
    printf "%-11s %s  %s / %s GiB used, %s GiB free\n" "$mnt" "$(bar $(( used*100/size )))" \
      "$(awk "BEGIN{printf \"%.1f\",$used/2^30}")" "$(awk "BEGIN{printf \"%.1f\",$size/2^30}")" "$(awk "BEGIN{printf \"%.1f\",$avail/2^30}")"
  done
  [ -d "$WS/logs" ] && printf "logs        %s\n" "$(du -sh "$WS/logs" 2>/dev/null | cut -f1)"
  if [ "$DETAIL" = 1 ] && [ -d "$WS" ]; then
    echo "Per folder:"; du -sh "$WS"/* 2>/dev/null | sort -rh | head -10 | sed 's/^/  /'
  fi

  hdr "Services"
  for s in $SERVICES; do
    name=${s%%:*}; port=${s##*:}
    res=$(curl -s -o /dev/null -m 3 -w '%{http_code} %{time_total}' "localhost:$port/health")
    code=${res%% *}; t=${res##* }
    case "$code" in
      200) st="${G}UP${N}      " ;;
      503) st="${Y}LOADING${N} " ;;
      000) st="${R}DOWN${N}    " ;;
      *)   st="${Y}HTTP $code${N}" ;;
    esac
    up=$(ps -eo etime=,args= 2>/dev/null | grep -E -- "--port $port( |$)" | grep -v grep | awk '{print $1; exit}')
    printf "%-4s :%-5s %b  health %3s ms   uptime %s\n" "$name" "$port" "$st" \
      "$(awk "BEGIN{printf \"%d\",$t*1000}")" "${up:--}"
  done

  hdr "Model load (vLLM)"
  for s in llm:8000 stt:8001; do
    name=${s%%:*}; port=${s##*:}
    run=$(metric "$port" vllm:num_requests_running); wait_=$(metric "$port" vllm:num_requests_waiting)
    kv=$(metric "$port" vllm:kv_cache_usage_perc vllm:gpu_cache_usage_perc)
    if [ -z "$run" ]; then printf "%-4s no metrics (service down?)\n" "$name"; continue; fi
    if [ -z "$kv" ]; then
      printf "%-4s running %-3s waiting %-3s KV cache n/a (metric not exposed)\n" "$name" "${run%.*}" "${wait_%.*}"
      continue
    fi
    # bar needs a whole number; round any non-zero usage up to at least 1% so activity is visible
    kvp=$(awk -v k="$kv" 'BEGIN{p=k*100; i=int(p); if (p>i) i++; print i}')
    kvd=$(awk -v k="$kv" 'BEGIN{printf "%.2f", k*100}')
    printf "%-4s running %-3s waiting %-3s KV cache %s  (%s%% exact)\n" "$name" "${run%.*}" "${wait_%.*}" "$(bar $kvp)" "$kvd"
  done
  echo "(KV cache is only used while requests run, so 0% when idle is normal; waiting > 0 means queuing)"
}

if [ "$WATCH" = 1 ]; then
  trap 'printf "\n"; exit 0' INT
  while true; do out=$(snapshot); clear; printf "%s\n" "$out"; printf "\nrefresh every %ss, Ctrl+C to stop\n" "$INTERVAL"; sleep "$INTERVAL"; done
else
  snapshot
fi