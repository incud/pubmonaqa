fixed_safety=$1

g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math \
    -fno-math-errno -funroll-loops -DNDEBUG \
    generation_annealing_schedules.cpp -o generation_annealing_schedules.x

mkdir -p schedule_outputs_fixed_safety
find schedule_outputs_fixed_safety -type f -name "*.txt" -empty -delete
echo "$fixed_safety" > schedule_outputs_fixed_safety/chosen_fixed_safety.txt

set -euo pipefail

for n in $(seq 5 30); do
    for idx in $(seq 0 99); do
        in="ising_inputs/model_n${n}_idx${idx}.txt"
        out="schedule_outputs_fixed_safety/schedule_n${n}_idx${idx}.txt"
        tmp="${out}.tmp"

        if [ -f "$out" ]; then
            echo "file <$in>: skipped because present"
            continue
        fi

        echo "file <$in>: processing"
        ./generation_annealing_schedules.x "$n" "$in" --fixed-safety "$fixed_safety" > "$tmp"
        mv "$tmp" "$out"
    done
done
