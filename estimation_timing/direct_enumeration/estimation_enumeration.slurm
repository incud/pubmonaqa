#!/bin/bash
#SBATCH --job-name=enumeration
#SBATCH --account=
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --output=log_enumeration_%j.out
#SBATCH --error=log_enumeration_%j.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

SOURCE_NAME=estimation_enumeration.cpp
EXEC_NAME=estimation_enumeration.x
ANALYSIS_NAME=collect_enumeration_timing.py

g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math -fno-math-errno -funroll-loops -DNDEBUG "$SOURCE_NAME" -o "$EXEC_NAME" 

mkdir -p enumeration_output

N_MIN=${N_MIN:-5}
N_MAX=${N_MAX:-30}
IDX_MIN=${IDX_MIN:-0}
IDX_MAX=${IDX_MAX:-9}

for idx in $(seq "$IDX_MIN" "$IDX_MAX"); do
    for n in $(seq "$N_MIN" "$N_MAX"); do
        in="ising_inputs/model_n${n}_idx${idx}.txt"
        out="enumeration_output/enumeration_n${n}_idx${idx}.txt"

        if [ ! -f "$in" ]; then
            echo "skipping n=$n idx=$idx because $in does not exist"
            continue
        fi

        if [ -f "$out" ] && grep -q '^completed 1$' "$out"; then
            echo "skipping n=$n idx=$idx because $out is already complete"
            continue
        fi

        echo "running n=$n idx=$idx"
        tmp="${out}.tmp.$$"
        ./$EXEC_NAME "$n" "$in" > "$tmp"
        mv "$tmp" "$out"
    done
done

python3 "$ANALYSIS_NAME" --input-dir enumeration_output --output estimation_timing.pkl
