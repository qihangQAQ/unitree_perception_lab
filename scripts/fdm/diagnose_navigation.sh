#!/usr/bin/env bash
# Read-only diagnostics. Run from the repository root while navigation is slow.
# Usage: bash scripts/fdm/diagnose_navigation.sh [RUN_DIR] [KIT_LOG]
#        Pipe to tee /tmp/fdm_diagnosis.txt to save the report.
set -u

fdm_run_dir=${1:-}
fdm_kit_log=${2:-}
if [[ -z "$fdm_run_dir" ]]; then
    fdm_run_dir=$(find logs/fdm_g1 -type f -path '*/stage3*/*' -name config.json \
        -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)
fi

fdm_progress() {
    date -Is
    if [[ -d "$fdm_run_dir" ]]; then
        find "$fdm_run_dir" -type f -name trajectory.jsonl -print -exec tail -n 1 {} \;
    fi
}

printf 'Navigation output: %s\n' "${fdm_run_dir:-not found; pass RUN_DIR explicitly}"
printf '\nGPU snapshot and memory\n'
nvidia-smi
free -h
printf '\nProcesses with highest CPU use\n'
ps -eo pid,etime,pcpu,pmem,rss,stat,args --sort=-pcpu | head -n 18

printf '\nNavigation progress before GPU sampling\n'
fdm_progress
printf '\nGPU samples (power, utilization, clocks, memory; approximately 10 seconds)\n'
nvidia-smi dmon -s pucm -d 1 -c 10
printf '\nNavigation progress after GPU sampling\n'
fdm_progress

if [[ -d "$fdm_run_dir" ]]; then
    printf '\nRun configuration\n'
    cat "$fdm_run_dir/config.json"
    printf '\nResults and recent events (missing results can mean the run is still in progress)\n'
    find "$fdm_run_dir" -type f \( -name summary.json -o -name result.json -o -name failure.json \) \
        -print -exec cat {} \;
    find "$fdm_run_dir" -type f -name events.jsonl -print -exec tail -n 12 {} \;
fi

if [[ -z "$fdm_kit_log" ]]; then
    printf '\nLocating the latest Kit log...\n'
    # ISAAC_PATH covers standalone installs, including a symlink outside HOME.
    fdm_log_roots=("$HOME" /opt /tmp)
    if [[ -n "${ISAAC_PATH:-}" ]]; then
        fdm_log_roots+=("$ISAAC_PATH/kit/logs")
    fi
    fdm_kit_log=$(find -H "${fdm_log_roots[@]}" -type f -name 'kit_*.log' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)
fi
if [[ -f "$fdm_kit_log" ]]; then
    printf '\nKit log: %s\n' "$fdm_kit_log"
    printf '\nErrors and navigation messages\n'
    fdm_pattern='\[Error\]|out of.*memory|Traceback|ImportError|CUDA error|CXXABI|GLIBCXX|\[FDM\]|Hang detected'
    if command -v rg >/dev/null 2>&1; then
        rg -ni "$fdm_pattern" "$fdm_kit_log" | tail -n 100
    else
        grep -niE "$fdm_pattern" "$fdm_kit_log" | tail -n 100
    fi
    printf '\nLast 200 Kit log lines\n'
    tail -n 200 "$fdm_kit_log"
else
    printf '\nKit log not found. Pass its path from the simulator startup message as the second argument.\n'
fi
