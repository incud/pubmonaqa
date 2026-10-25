# FPGA Metropolis arithmetic

All run scripts take no parameters. The hardcoded settings are beta 1, degree 12,
error budgets 1e-4, test size 10, synthesis sizes 8/16/32/64, Vitis 2025.2,
xcvu19p-fsva3824-2-e, and target clock 3 ns. Edit `scripts/run_phase.sh`
if needed. Requirements: Python 3 with NumPy/SciPy, g++, and Vitis with a license.

Run the phases separately:

```bash
bash run_all_generate.sh
bash run_all_wallace.sh
bash run_all_test_local.sh
bash run_all_test_uniform.sh
bash run_8_local.sh
bash run_8_uniform.sh
bash run_16_local.sh
bash run_16_uniform.sh
bash run_32_local.sh
bash run_32_uniform.sh
bash run_64_local.sh
bash run_64_uniform.sh
bash run_all_reports.sh
```

The two move-test scripts each run their testbench at n=10,8,16,32,64, retaining
coverage of every configured precision. Each `run_<n>_<move>.sh` performs exactly
one synthesis. Header generation must finish before tests or synthesis; the
report script runs after all eight syntheses. It copies XML/text reports, prints
the eight latencies, and writes `reports/summary.csv`. Shared settings and actions
are in `scripts/run_phase.sh`. `run.sh` is an optional wrapper for this sequence.
Failures stop the current script. Output goes to the terminal; tool journals/logs
are removed only from that script's job directories, including on failure.

The Tcl driver uses `open_project`/`open_solution` for C simulation, explicitly
compiling the kernel and testbench together. Synthesis uses `v++ -c --mode hls`
with absolute source paths in a generated config, avoiding Tcl source-path
rewriting. Each generated project is recreated before its run.

## Physical energies

Inputs are the physical coefficients of
`H(s) = -sum h_i s_i - sum J_ij s_i s_j`, with `sum c_l^2 = n` before rounding.
No hardware energy normalization is used. The fractional precision is
`b = ceil(log2(beta*M / log(1 + 2*eps_b)))`, where `M = n*(n+1)/2`.
The signed integer part represents `B_n = 2*ceil(n*sqrt(n)/sqrt(pi))`, including
positive B_n. At n=8/16/32/64 the `(fractional, integer, total)` widths are
`(18,6,24)`, `(20,8,28)`, `(22,9,31)`, `(24,11,35)`.

The generator checks the range using the SK normalization: a k-spin change
affects k(n+1-k) coefficients, so Cauchy-Schwarz bounds the rounded difference by
`2*sqrt(n*floor((n+1)^2/4)) + floor((n+1)^2/4)*2^-b`.
It also checks the range of energy partial sums.

## Wallace tree and exponential

`src/wallace.hpp` contains carry-save 3:2 compressors and a parallel-prefix final
adder: O(log N + log W) logical depth for N terms of W bits. The same template
works with HLS integer words and standard unsigned C++ words. There is no
`wallace_probe` wrapper: `tb/tb_wallace.cpp` calls the template directly and
compiles with g++, without Vitis headers. It checks 131072 byte addition/carry-in
cases and 2640 multioperand cases, including full 2080-term sums.

Acceptance is exactly one for nonpositive energy differences. Positive
differences below `log(2/eps_op)/beta` use a degree-12 Chebyshev polynomial
evaluated by Clenshaw recurrence and clipped to [0,1]. At or above the cutoff,
acceptance is zero. The cutoff is rounded upward to the energy grid, so the
discarded exact probability is at most `eps_op/2 = 5e-5`. For beta 1, the cutoff
is approximately 9.90349. Hamiltonian coefficients and energies stay in physical
units; no repeated-squaring stages are used.

The move tests compare rounded-grid energies against independent dense-energy
calculations and check acceptance over the full range, near zero, and at the
cutoff and its adjacent energy-grid values. The local
kernel receives a one-hot flip mask. The uniform kernel receives the candidate
and cached current energy. Both return a signed difference and an approximate
acceptance probability; proposal generation, random comparison and state updates
belong to the caller. Per-evaluation acceptance accuracy does not establish an
end-to-end TV bound or exact detailed balance.

## Reports

`scripts/reports.py` extracts just `Worst-caseRealTimeLatency` from each top-level
XML and converts its units to seconds. `reports/summary.csv` contains only
`move,n,latency_s`. No fitting or resource/clock/precision parsing is performed.
The latency is the nominal HLS estimate for the arithmetic kernel; placement,
routing, initiation interval and the state-update loop are outside this value.
Run Vitis again for the revised circuits before updating the paper's timings.
