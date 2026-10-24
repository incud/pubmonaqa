#include "fpga.hpp"

void local_spin_flip_operation(const coeff_t h[SK_N], const coeff_t J[NUM_J],
    ap_uint<SK_N> spins, ap_uint<SK_N> flip_mask,
    energy_t* delta_energy, prob_t* accept_prob) {
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=J complete dim=1
#pragma HLS INTERFACE ap_none port=spins
#pragma HLS INTERFACE ap_none port=flip_mask
#pragma HLS INTERFACE ap_none port=delta_energy
#pragma HLS INTERFACE ap_none port=accept_prob
#pragma HLS INTERFACE ap_ctrl_none port=return
    // The caller supplies a one-hot mask and keeps coefficients resident on chip.
    energy_t field = 0;
    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        if (flip_mask[i]) field = h[i];
    }
    energy_t terms[SK_N + 1];
#pragma HLS ARRAY_PARTITION variable=terms complete dim=1
    terms[0] = field;
    for (int j = 0; j < SK_N; ++j) {
#pragma HLS UNROLL
        energy_t coupling = 0;
        for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
            if (i != j && flip_mask[i]) coupling = J[edge_index(i, j)];
        }
        terms[j + 1] = signed_term(coupling, !bool(spins[j]));
    }
    energy_t local_field = sum_energy<SK_N + 1>(terms);
    bool selected_positive = ((spins & flip_mask) != 0);
    energy_t signed_field = signed_term(local_field, !selected_positive);
    energy_t delta = energy_from_bits(ap_int<ENERGY_BITS>(energy_bits(signed_field) << 1));
    *delta_energy = delta;
    *accept_prob = metropolis_probability(delta);
}
