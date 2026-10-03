#!/bin/bash
# Live GPU VM dashboard: GPU, CPU, RAM, disk, services, LLM/STT load and token throughput.
# Usage: status.sh               live view, refreshes every 1 s (Ctrl+C to exit)
#        status.sh -i 2          live view, refresh every 2 s
#        status.sh --once        print one snapshot and exit
WS=${WS:-/workspace}
INTERVAL=1; ONCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    -i) INTERVAL=$2; shift ;;
    --once|-1) ONCE=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
  esac; shift
done

B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; C=$'\e[36m'; N=$'\e[0m'
SPARK=(▁ ▂ ▃ ▄ ▅ ▆ ▇ █); HIST=30

col() { local p=${1%.*}; [ -z "$p" ] && p=0
  if [ "$p" -ge 90 ]; then printf %s "$R"; elif [ "$p" -ge 75 ]; then printf %s "$Y"; else printf %s "$G"; fi; }
bar() { local p=${1%.*}; [ -z "$p" ] && p=0; [ "$p" -gt 100 ] && p=100; local f=$((p/5))
  printf "%s[" "$(col "$p")"; printf "%${f}s" "" | tr ' ' '#'; printf "%$((20-f))s" "" | tr ' ' '.'; printf "]%s %3s%%" "$N" "$p"; }
# spark <max> <values...>  ->  sparkline of the values scaled to max
spark() { local max=$1; shift; local out="" v i
  for v in "$@"; do i=$(awk -v v="$v" -v m="$max" 'BEGIN{ if (m<=0) {print 0; exit} i=int(v/m*7+0.5); if (i>7) i=7; if (i<0) i=0; print i }')
    out+="${SPARK[$i]}"; done; printf "%s%s%s" "$C" "$out" "$N"; }
push() { local -n arr=$1; arr+=("$2"); [ "${#arr[@]}" -gt "$HIST" ] && arr=("${arr[@]:1}"); }
f1() { awk -v x="$1" 'BEGIN{printf "%.1f", x}'; }

# metrics <port>: prints key=value for the vLLM metrics we use (sums over label sets)
metrics() { curl -s -m 1 "localhost:$1/metrics" | awk '
  /^vllm:/ { n=$1; sub(/\{.*/, "", n); v[n]+=$NF; if (!model && match($0, /model_name="[^"]*"/)) model=substr($0, RSTART+12, RLENGTH-13) }
  END { for (k in v) print k "=" v[k]; if (model) print "model=" model }'; }
mget() { printf "%s\n" "$1" | awk -F= -v k="$2" '$1==k {print $2; exit}'; }

# CPU % from /proc/stat deltas between frames (no sleeping)
read_cpu() { read -r _ a b c d e f g h _ < /proc/stat; CPU_T=$((a+b+c+d+e+f+g+h)); CPU_I=$((d+e)); }
read_cpu; PREV_T=$CPU_T; PREV_I=$CPU_I; sleep 0.3   # short baseline so the first CPU reading is real

declare -a H_UTIL=() H_GEN=()
PREV_GEN=""; PREV_PROMPT=""; PREV_REQ=""; PREV_TS=""
DISK_CACHE=""; DISK_TS=0

frame() {
  local now ts out; now=$(date '+%Y-%m-%d %H:%M:%S'); ts=$(date +%s.%N)
  out="${B}GPU VM live${N}  $now  ${D}$(hostname)${N}\n"

  # --- GPU ---
  out+="\n${B}GPU${N}\n"
  if command -v nvidia-smi >/dev/null; then
    IFS=, read -r name used total util temp pw pl < <(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)
    used=${used// /}; total=${total// /}; util=${util// /}
    push H_UTIL "${util:-0}"
    out+="  $(echo $name)   ${temp// /} C   $(echo ${pw%.*}) / $(echo ${pl%.*}) W\n"
    out+="  VRAM  $(bar $(( used*100/total )))  $(f1 "$(awk "BEGIN{print $used/1024}")") / $(f1 "$(awk "BEGIN{print $total/1024}")") GiB\n"
    out+="  Util  $(bar "$util")  $(spark 100 "${H_UTIL[@]}")\n"
  else out+="  nvidia-smi not found\n"; fi

  # --- CPU / RAM ---
  read_cpu; local dt=$((CPU_T-PREV_T)) di=$((CPU_I-PREV_I)) cpu=0
  [ "$dt" -gt 0 ] && cpu=$(( 100*(dt-di)/dt )); PREV_T=$CPU_T; PREV_I=$CPU_I
  read -r mt mu < <(free -b | awk '/^Mem:/{print $2,$3}')
  out+="\n${B}Host${N}\n"
  out+="  CPU   $(bar $cpu)  $(nproc) vCPU\n"
  out+="  RAM   $(bar $(( mu*100/mt )))  $(f1 "$(awk "BEGIN{print $mu/2^30}")") / $(f1 "$(awk "BEGIN{print $mt/2^30}")") GiB\n"
  # disk changes slowly: refresh every 30 s
  if [ $(( ${ts%.*} - DISK_TS )) -ge 30 ] || [ -z "$DISK_CACHE" ]; then
    DISK_CACHE=$(df -P -B1 "$WS" 2>/dev/null | awk 'NR==2{print $2,$3}'); DISK_TS=${ts%.*}; fi
  read -r ds du_ <<< "$DISK_CACHE"
  [ -n "$ds" ] && out+="  Disk  $(bar $(( du_*100/ds )))  $(f1 "$(awk "BEGIN{print $du_/2^30}")") / $(f1 "$(awk "BEGIN{print $ds/2^30}")") GiB  ($WS)\n"

  # --- services ---
  out+="\n${B}Services${N}\n"
  local s name port code st up
  for s in llm:8000 stt:8001 tts:8002; do
    name=${s%%:*}; port=${s##*:}
    code=$(curl -s -o /dev/null -m 1 -w '%{http_code}' "localhost:$port/health")
    case "$code" in 200) st="${G}● UP     ${N}";; 503) st="${Y}● LOADING${N}";; *) st="${R}● DOWN   ${N}";; esac
    up=$(ps -eo etime=,args= 2>/dev/null | grep -E -- "--port $port( |$)" | grep -v grep | awk '{print $1; exit}')
    out+="  $(printf '%-4s' $name) :$port  $st  uptime ${up:--}\n"
  done

  # --- LLM load + throughput ---
  local m run wait kv gen prompt req model gr pr rr
  m=$(metrics 8000)
  out+="\n${B}LLM${N}"
  if [ -n "$m" ]; then
    model=$(mget "$m" model); run=$(mget "$m" vllm:num_requests_running); wait=$(mget "$m" vllm:num_requests_waiting)
    kv=$(mget "$m" vllm:kv_cache_usage_perc); [ -z "$kv" ] && kv=$(mget "$m" vllm:gpu_cache_usage_perc)
    gen=$(mget "$m" vllm:generation_tokens_total); prompt=$(mget "$m" vllm:prompt_tokens_total); req=$(mget "$m" vllm:request_success_total)
    gr=0; pr=0; rr=0
    if [ -n "$PREV_TS" ] && [ -n "$gen" ] && [ -n "$PREV_GEN" ]; then
      gr=$(awk -v a="$gen" -v b="$PREV_GEN" -v t="$ts" -v u="$PREV_TS" 'BEGIN{d=t-u; if (d<=0||a<b) print 0; else printf "%.0f", (a-b)/d}')
      pr=$(awk -v a="$prompt" -v b="$PREV_PROMPT" -v t="$ts" -v u="$PREV_TS" 'BEGIN{d=t-u; if (d<=0||a<b) print 0; else printf "%.0f", (a-b)/d}')
      rr=$(awk -v a="${req:-0}" -v b="${PREV_REQ:-0}" 'BEGIN{ if (a<b) print 0; else printf "%.0f", a-b}')
    fi
    PREV_GEN=$gen; PREV_PROMPT=$prompt; PREV_REQ=$req; PREV_TS=$ts
    push H_GEN "$gr"
    local gmax; gmax=$(printf "%s\n" "${H_GEN[@]}" | sort -n | tail -1); [ "${gmax:-0}" -lt 50 ] && gmax=50
    out+="  ${D}${model:-?}${N}\n"
    out+="  Requests  running ${B}${run%.*}${N}  waiting $( [ "${wait%.*}" -gt 0 ] 2>/dev/null && printf %s "$Y${wait%.*}$N" || printf %s "${wait%.*}")  done total ${req%.*}"
    [ "$rr" -gt 0 ] 2>/dev/null && out+="  ${G}+$rr${N}"
    out+="\n"
    if [ -n "$kv" ]; then
      local kvp kvd; kvp=$(awk -v k="$kv" 'BEGIN{p=k*100; i=int(p); if (p>i) i++; print i}'); kvd=$(awk -v k="$kv" 'BEGIN{printf "%.2f", k*100}')
      out+="  KV cache  $(bar "$kvp")  ${D}${kvd}% exact${N}\n"
    else out+="  KV cache  n/a\n"; fi
    out+="  Tokens/s  generate ${B}$(printf '%5s' "$gr")${N}   prompt $(printf '%6s' "$pr")   $(spark "$gmax" "${H_GEN[@]}")\n"
  else out+="\n  no metrics (down or starting)\n"; PREV_TS=""; fi

  # --- STT ---
  m=$(metrics 8001)
  out+="\n${B}STT${N}"
  if [ -n "$m" ]; then
    out+="  ${D}$(mget "$m" model)${N}\n  Requests  running $(mget "$m" vllm:num_requests_running | cut -d. -f1)  waiting $(mget "$m" vllm:num_requests_waiting | cut -d. -f1)  done total $(mget "$m" vllm:request_success_total | cut -d. -f1)\n"
  else out+="\n  no metrics (down or starting)\n"; fi

  [ "$ONCE" = 0 ] && out+="\n${D}refresh ${INTERVAL}s · Ctrl+C to exit · status.sh --once for a single snapshot${N}\n"
  FRAME="$out"
}

if [ "$ONCE" = 1 ]; then frame; printf "%b" "$FRAME"; exit 0; fi
trap 'printf "\e[?25h\n"; exit 0' INT TERM
printf "\e[?25l\e[2J"   # hide cursor, clear once
while true; do
  frame
  printf "\e[H%b\e[J" "$FRAME"   # redraw in place: no flicker
  sleep "$INTERVAL"
done