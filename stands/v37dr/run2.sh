#!/bin/bash
# usage: [DSX=dir] run2.sh <estimator> <tag> <n_mlps> [env...]; охорона RAM: якщо MemAvailable < 4 ГБ, прогін вбивається
DS=${DSX:-/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$1; T=$2; N=$3; shift 3
env "$@" nice -n 10 ${WB:-/home/trim/fs16}/bin/whest run --estimator $E --dataset $DS --split mini --n-mlps $N --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/v37_$T.json 2>/home/trim/wblogs/v37_$T.err &
P=$!; PK=0
while kill -0 $P 2>/dev/null; do
  A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  R=$(ps -o rss= --ppid $P -p $P 2>/dev/null | awk '{s+=$1} END{print int(s/1048576)}')
  [ "$R" -gt "$PK" ] && PK=$R
  if [ "$A" -lt 4 ]; then pkill -P $P; kill $P; echo "OOMGUARD $T avail=${A}G rss=${R}G" >> /home/trim/wblogs/v37_runs.log; fi
  sleep 5
done
echo "DONE $T peakRSS=${PK}G" >> /home/trim/wblogs/v37_runs.log
