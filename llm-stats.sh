#!/bin/bash
# LLM usage stats from vLLM's /metrics (totals since the server started).
# Usage: llm-stats.sh            stats for the LLM (port 8000)
#        llm-stats.sh -p 8001    stats for another vLLM server (e.g. STT)
#        llm-stats.sh -w [N]     refresh every N seconds (default 5)
#        llm-stats.sh --raw      list every vllm: metric name (to discover more)
PORT=8000; WATCH=0; INTERVAL=5; RAW=0
while [ $# -gt 0 ]; do
  case "$1" in
    -p) PORT=$2; shift ;;
    -w) WATCH=1; [[ "${2:-}" =~ ^[0-9]+$ ]] && { INTERVAL=$2; shift; } ;;
    --raw) RAW=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
  esac; shift
done

fetch() { curl -s -m 3 "localhost:$PORT/metrics"; }
# m <name...>: sum of the first metric name that exists (all label sets added together)
m() { for n in "$@"; do
        v=$(printf "%s\n" "$DATA" | awk -v n="$n" '$1==n || index($1, n"{")==1 {s+=$NF; f=1} END{if(f) printf "%.6f", s}')
        [ -n "$v" ] && { echo "$v"; return; }
      done; echo ""; }
div() { awk -v a="$1" -v b="$2" -v f="$3" 'BEGIN{ if (a=="" || b=="" || b+0==0) print "n/a"; else printf f, a/b }'; }
int() { [ -z "$1" ] && echo "n/a" || awk -v x="$1" 'BEGIN{printf "%d", x}'; }

report() {
  DATA=$(fetch)
  if [ -z "$DATA" ]; then echo "No metrics on port $PORT (server down or starting?)"; return; fi
  if [ "$RAW" = 1 ]; then printf "%s\n" "$DATA" | grep -E '^vllm:' | sed 's/{.*//' | sort -u; return; fi

  model=$(printf "%s\n" "$DATA" | grep -o 'model_name="[^"]*"' | head -1 | cut -d'"' -f2)
  req_ok=$(m vllm:request_success_total vllm:request_success)
  p_tok=$(m vllm:prompt_tokens_total vllm:prompt_tokens)
  g_tok=$(m vllm:generation_tokens_total vllm:generation_tokens)
  ttft_s=$(m vllm:time_to_first_token_seconds_sum); ttft_c=$(m vllm:time_to_first_token_seconds_count)
  e2e_s=$(m vllm:e2e_request_latency_seconds_sum);  e2e_c=$(m vllm:e2e_request_latency_seconds_count)
  itl_s=$(m vllm:inter_token_latency_seconds_sum vllm:time_per_output_token_seconds_sum)
  itl_c=$(m vllm:inter_token_latency_seconds_count vllm:time_per_output_token_seconds_count)
  pc_q=$(m vllm:prefix_cache_queries_total vllm:prefix_cache_queries)
  pc_h=$(m vllm:prefix_cache_hits_total vllm:prefix_cache_hits)
  run=$(m vllm:num_requests_running); wait_=$(m vllm:num_requests_waiting)
  kv=$(m vllm:kv_cache_usage_perc vllm:gpu_cache_usage_perc)

  printf "\e[1mvLLM stats\e[0m  port %s  model %s  %s\n" "$PORT" "${model:-?}" "$(date '+%H:%M:%S')"
  echo "-- Totals since server start --"
  printf "  Requests completed   %s\n" "$(int "$req_ok")"
  printf "  Prompt tokens in     %s\n" "$(int "$p_tok")"
  printf "  Tokens generated     %s\n" "$(int "$g_tok")"
  echo "-- Averages per request --"
  printf "  Prompt tokens        %s\n" "$(div "$p_tok" "$req_ok" "%.0f")"
  printf "  Generated tokens     %s\n" "$(div "$g_tok" "$req_ok" "%.0f")"
  printf "  Time to first token  %s s\n" "$(div "$ttft_s" "$ttft_c" "%.2f")"
  printf "  Total request time   %s s\n" "$(div "$e2e_s" "$e2e_c" "%.2f")"
  printf "  Generation speed     %s tokens/s per request\n" "$(div 1 "$(div "$itl_s" "$itl_c" "%.6f" | sed 's/n\/a//')" "%.0f")"
  printf "  Prefix cache hits    %s %%\n" "$(div "$(awk -v h="${pc_h:-0}" 'BEGIN{print h*100}')" "$pc_q" "%.0f")"
  echo "-- Right now --"
  printf "  Running %s  Waiting %s  KV cache %s %%\n" "$(int "$run")" "$(int "$wait_")" \
    "$( [ -n "$kv" ] && awk -v k="$kv" 'BEGIN{printf "%.0f", k*100}' || echo n/a)"
}

if [ "$WATCH" = 1 ]; then
  trap 'echo; exit 0' INT
  while true; do out=$(report); clear; printf "%s\n\nrefresh every %ss, Ctrl+C to stop\n" "$out" "$INTERVAL"; sleep "$INTERVAL"; done
else report; fi
