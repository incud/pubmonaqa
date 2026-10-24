#include "fpga.hpp"

void uniform_move_operation(const coeff_t h[SK_N], const coeff_t J[NUM_J],
    ap_uint<SK_N> proposed_spins, energy_t old_energy,
    energy_t* new_energy, energy_t* delta_energy, prob_t* accept_prob) {
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=J complete dim=1
#pragma HLS INTERFACE ap_none port=proposed_spins
#pragma HLS INTERFACE ap_none port=old_energy
#pragma HLS INTERFACE ap_none port=new_energy
#pragma HLS INTERFACE ap_none port=delta_energy
#pragma HLS INTERFACE ap_none port=accept_prob
#pragma HLS INTERFACE ap_ctrl_none port=return
    energy_t terms[NUM_TERMS];
#pragma HLS ARRAY_PARTITION variable=terms complete dim=1
    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        terms[i] = signed_term(h[i], bool(proposed_spins[i]));
    }
    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        for (int j = i + 1; j < SK_N; ++j) {
#pragma HLS UNROLL
            int e = edge_index(i, j);
            terms[SK_N + e] = signed_term(J[e], proposed_spins[i] == proposed_spins[j]);
        }
    }
    energy_t energy = sum_energy<NUM_TERMS>(terms);
    energy_t delta = energy_add(energy, old_energy, true);
    *new_energy = energy;
    *delta_energy = delta;
    *accept_prob = metropolis_probability(delta);
}
