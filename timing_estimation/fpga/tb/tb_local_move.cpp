#include <ap_fixed.h>
#include <ap_int.h>
#include <cmath>
#include <iostream>

#ifndef SK_N
#error "SK_N is not defined."
#endif

#ifndef F
#error "F is not defined."
#endif

static const int NUM_J = SK_N * (SK_N - 1) / 2;

using coeff_t = ap_fixed<F + 1, 1, AP_RND, AP_SAT>;
using delta_t = ap_fixed<F + 1, 1, AP_RND, AP_SAT>;
using prob_t = ap_ufixed<32, 1, AP_RND, AP_SAT>;

void local_spin_flip_operation(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> spins,
    ap_uint<SK_N> flip_mask,
    delta_t* delta_x,
    prob_t* accept_prob
);

static int edge_index_ref(int i, int j) {
    int a = i < j ? i : j;
    int b = i < j ? j : i;
    return a * (2 * SK_N - a - 1) / 2 + (b - a - 1);
}

static double spin_ref(bool b) {
    return b ? 1.0 : -1.0;
}

int main() {
    coeff_t h[SK_N];
    coeff_t J_edges[NUM_J];

    double hd[SK_N];
    double Jd[NUM_J];

    for (int i = 0; i < SK_N; ++i) {
        hd[i] = 1e-3 * double((i % 5) - 2);
        h[i] = coeff_t(hd[i]);
    }

    for (int e = 0; e < NUM_J; ++e) {
        Jd[e] = 1e-4 * double((e % 7) - 3);
        J_edges[e] = coeff_t(Jd[e]);
    }

    ap_uint<SK_N> spins = 0;
    for (int i = 0; i < SK_N; ++i) {
        spins[i] = (i % 2 == 0);
    }

    int flip_i = SK_N / 3;
    ap_uint<SK_N> flip_mask = 0;
    flip_mask[flip_i] = 1;

    delta_t delta_x;
    prob_t accept_prob;

    local_spin_flip_operation(h, J_edges, spins, flip_mask, &delta_x, &accept_prob);

    double field = hd[flip_i];
    for (int j = 0; j < SK_N; ++j) {
        if (j != flip_i) {
            field += Jd[edge_index_ref(flip_i, j)] * spin_ref(bool(spins[j]));
        }
    }

    double raw_delta = 2.0 * spin_ref(bool(spins[flip_i])) * field;
    double ref_delta_x = raw_delta > 0.0 ? raw_delta : 0.0;

    double got_delta_x = delta_x.to_double();
    double got_prob = accept_prob.to_double();
    double err = std::abs(got_delta_x - ref_delta_x);

    std::cout << "SK_N " << SK_N << "\n";
    std::cout << "F " << F << "\n";
    std::cout << "flip_i " << flip_i << "\n";
    std::cout << "ref_delta_x " << ref_delta_x << "\n";
    std::cout << "got_delta_x " << got_delta_x << "\n";
    std::cout << "abs_error " << err << "\n";
    std::cout << "accept_prob " << got_prob << "\n";

    if (err > 4.0 * std::ldexp(1.0, -F)) {
        std::cerr << "ERROR: delta_x mismatch too large\n";
        return 1;
    }

    if (got_prob < 0.0 || got_prob > 1.0) {
        std::cerr << "ERROR: probability outside [0,1]\n";
        return 1;
    }

    return 0;
}
