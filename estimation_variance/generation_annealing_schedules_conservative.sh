g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math \
    -fno-math-errno -funroll-loops -DNDEBUG \
    generation_annealing_schedules.cpp -o generation_annealing_schedules_conservative.x

mkdir -p schedule_outputs_conservative

set -euo pipefail

for idx in $(seq 0 99); do
    for n in $(seq 5 30); do
        in="ising_inputs/model_n${n}_idx${idx}.txt"
        out="schedule_outputs_conservative/schedule_n${n}_idx${idx}.txt"
        tmp="${out}.tmp"

        if [ -f "$out" ]; then
            echo "file <$in>: skipped because present"
            continue
        fi

        echo "file <$in>: processing"
        ./generation_annealing_schedules_conservative.x "$n" "$in" --conservative-fit > "$tmp"
        mv "$tmp" "$out"
    done
done
