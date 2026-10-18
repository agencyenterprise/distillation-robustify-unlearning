#!/bin/bash
#SBATCH --job-name=agi2569-warm
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Builds (or just loads) the datasets caches a stage's runs read, once, before its array fans out: file locks do
# not work across nodes, and concurrent first builds crashed calibration runs. CPU only (NUM_PROC=24 workers).
# Arguments: the stage's script, then the sweep arguments of the runs to come, e.g.
#   sbatch cluster/agi2569/warm_cache.sbatch run_partial_distill_language.py --arms B C D
#   sbatch cluster/agi2569/warm_cache.sbatch run_relearn_language.py
# The log ends with the number of cache files it built (0: every cache already existed).
source cluster/agi2569/env.sh
$PY "$1" --warm_cache "${@:2}"
