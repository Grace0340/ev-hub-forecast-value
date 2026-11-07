#!/usr/bin/env bash
# Run the pipeline in stages:  bash scripts/run_all.sh [stage ...]   (no argument: every stage in order)
# Stages: data forecast base sensitivity seeds nopeek costaware summary
# Environment: PY (Python interpreter), NPROC (worker processes), EV_HUB_ROOT (folder for data/ and outputs/),
#              CITIES (CHARGED cities, default "SZH LOA MEL").
set -euo pipefail
set -f   # forecast-file patterns are expanded by the Python scripts, not by the shell

REPO=$(cd "$(dirname "$0")/.." && pwd)
export EV_HUB_ROOT=${EV_HUB_ROOT:-$REPO}
export NPROC=${NPROC:-$(nproc)}
PY="${PY:-python} -W ignore"
LOG=$EV_HUB_ROOT/outputs/logs
mkdir -p "$LOG"
cd "$REPO/src"

CITIES=${CITIES:-SZH LOA MEL}
declare -A TAR=([uev]=sz [SZH]=sz [LOA]=sce [MEL]=vdo)
UOUT=$EV_HUB_ROOT/outputs/urbanev
# seed-0 forecast pool (the baselines carry seed 0), the seed-1/2 forecasts of the same models, and both together
P0="preds_A_*_mse_s0.npz preds_A_*_mae_s0.npz preds_A_*_pinball*_s0.npz preds_A_*_ph_s0.npz preds_A_*_cost_s0.npz"
P12="preds_A_*_mse_s[12].npz preds_A_*_mae_s[12].npz preds_A_*_pinball*_s[12].npz preds_A_*_ph_s[12].npz
     preds_A_*_cost_s[12].npz"
PALL="$P0 $P12"

job() {  # log name, then command
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] $name"
  if ! "$@" > "$LOG/$name.txt" 2>&1; then
    echo "failed: $name (see $LOG/$name.txt)" >&2
    exit 1
  fi
}

stage_data() {
  bash "$REPO/scripts/download_data.sh"
  job prep_urbanev env DATASET=uev $PY preprocess_urbanev.py
  job retail_urbanev env DATASET=uev $PY retail_urbanev.py
  for c in $CITIES; do job prep_$c env DATASET=$c $PY preprocess_charged.py; done
}

stage_forecast() {  # GPU recommended
  for ds in uev $CITIES; do
    job fc_${ds}_mse env DATASET=$ds $PY forecast.py --loss mse
    for loss in mae pinball70 pinball90; do
      job fc_${ds}_$loss env DATASET=$ds $PY forecast.py --models DLinear GRU iTransformer --loss $loss
    done
  done
  # UrbanEV pool only: post-hoc shift and cost loss with the preliminary weights
  job fc_uev_ph env DATASET=uev $PY posthoc_shift.py --suffix ph
  job fc_uev_cost env DATASET=uev $PY forecast.py --models DLinear GRU iTransformer --loss cost
}

stage_base() {  # both controllers, every dataset, calibration and test periods
  for ds in uev $CITIES; do
    local e="env DATASET=$ds TARIFF=${TAR[$ds]}"
    for p in test cal; do job da_${ds}_$p $e $PY cost_dayahead.py --period $p --tag fx_base --pattern $PALL; done
    job an_${ds}_da $e $PY analysis.py --tag fx_base
    for p in test cal; do job ro_${ds}_$p $e $PY rolling.py --period $p --tag fx_mpc --pattern $P0; done
    job an_${ds}_ro $e $PY analysis.py --tag fx_mpc
  done
}

case_da() {  # suffix, then env assignments, then --, then cost_dayahead.py options (UrbanEV, day-ahead only)
  local s=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  for p in test cal; do
    job da_${s}_$p env DATASET=uev TARIFF=sz ${envs[@]+"${envs[@]}"} $PY cost_dayahead.py --period $p --tag fx_$s \
      --pattern $P0 "$@"
  done
  job an_$s env DATASET=uev TARIFF=sz ${envs[@]+"${envs[@]}"} $PY analysis.py --tag fx_$s
}

case_both() {  # as case_da, plus rolling re-optimisation on the test period
  local s=$1
  case_da "$@"
  shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  job ro_$s env DATASET=uev TARIFF=sz ${envs[@]+"${envs[@]}"} $PY rolling.py --period test --tag ms_$s \
    --pattern $P0 "$@"
}

stage_sensitivity() {  # UrbanEV hub, tariff and valuation cases
  case_both pv0 -- --pv 0
  case_both pv50 -- --pv 0.5
  case_both b10 -- --batt 0.10
  case_both b50 -- --batt 0.50
  case_da g90 -- --gq 0.90
  case_da g99 -- --gq 0.99
  case_both gd TARIFF=gd --
  case_both szt TARIFF=szt --
  case_both deg25 DEG_COST=0.025 --
  case_both deg50 DEG_COST=0.050 --
  case_both v05 VOLL_MULT=0.5 --
  case_both v2 VOLL_MULT=2 --
  case_both tflat END_VALUE=flat --
}

stage_seeds() {
  job ro_seeds env DATASET=uev TARIFF=sz $PY rolling.py --period test --tag rb_seed --pattern $P12
}

stage_nopeek() {  # rolling re-optimisation that does not observe the current hour
  for ds in uev $CITIES; do
    job np_$ds env DATASET=$ds TARIFF=${TAR[$ds]} MPC_PEEK=0 $PY rolling.py --period test --tag np_mpc --pattern $P0
  done
}

stage_costaware() {  # controller-specific weights (UrbanEV): d = day-ahead following, m = rolling re-optimisation
  local e="env DATASET=uev TARIFF=sz"
  job cw_weights_d $e $PY cost_weights.py --tag fx_base
  job cw_weights_m $e $PY cost_weights.py --tag fx_mpc
  job cw_shift_d $e $PY posthoc_shift.py --weights "$UOUT/cost_weights_fx_base.json" --suffix phd
  job cw_shift_m $e $PY posthoc_shift.py --weights "$UOUT/cost_weights_fx_mpc.json" --suffix phm
  job cw_train_d $e $PY forecast.py --models DLinear GRU iTransformer --loss cost --name costd \
    --weights "$UOUT/cost_weights_fx_base.json"
  job cw_train_m $e $PY forecast.py --models DLinear GRU iTransformer --loss cost --name costm \
    --weights "$UOUT/cost_weights_fx_mpc.json"
  for kind in ph cost; do
    for p in test cal; do
      job cw_da_${kind}_$p $e $PY cost_dayahead.py --period $p --tag cw_da_$kind --pattern "preds_A_*_${kind}[dm]_s*.npz"
    done
    job cw_ro_$kind $e $PY rolling.py --period test --tag cw_mpc_$kind --pattern "preds_A_*_${kind}[dm]_s0.npz"
  done
}

stage_summary() {
  local e="env DATASET=uev TARIFF=sz"
  job sum_tables $e $PY summary_tables.py
  job sum_costaware $e $PY summary_costaware.py
  job sum_robust $e $PY summary_robust.py
  job sum_nopeek $e $PY summary_nopeek.py
  job sum_plain $e $PY summary_plain.py
  echo "summary tables in $EV_HUB_ROOT/outputs/summary"
}

STAGES=${*:-data forecast base sensitivity seeds nopeek costaware summary}
for s in $STAGES; do
  echo "== stage $s"
  "stage_$s"
done
