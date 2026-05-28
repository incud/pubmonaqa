/*
 * Testbench for the local single-spin-flip HLS kernel.
 *
 * This testbench fixes SK_N=10 and exhaustively scans all 2^10 spin
 * configurations. For each configuration, it tests all SK_N possible
 * one-hot local spin flips. Therefore the total number of checked moves is
 * 2^10 * 10.
 *
 * For each move, the testbench calls local_spin_flip_operation(...), which
 * computes x=max(Delta E_i/alpha,0) and the corresponding Metropolis
 * acceptance probability. The reference value of Delta E_i/alpha is computed
 * independently in double precision from
 *
 *     Delta E_i/alpha = 2 s_i (h_i/alpha + sum_{j != i} (J_ij/alpha) s_j).
 *
 * The testbench then compares:
 *
 *     1. the HLS output delta_x against the double-precision reference;
 *     2. the HLS output accept_prob against std::exp(-TEST_LAMBDA * delta_x),
 *        where TEST_LAMBDA = beta * alpha.
 *
 * The coefficients h and J_edges are deterministic small normalized values,
 * chosen so that all intermediate normalized energy differences remain safely
 * inside the intended fixed-point range. The packed upper-triangular indexing
 * of J_edges is also checked implicitly, since every spin index is flipped
 * across all spin configurations.
 *
 * The test prints all failing cases and a few sample of passing cases.
 */


#include <ap_fixed.h>
#include <ap_int.h>
#include <cmath>
#include <iomanip>
#include <iostream>

#ifndef SK_N
#error "SK_N is not defined."
#endif

#if SK_N != 10
#error "This testbench is intentionally for SK_N=10. Compile with -DSK_N=10."
#endif

#ifndef FRAC
#error "FRAC is not defined."
#endif

#ifndef TEST_LAMBDA
#error "TEST_LAMBDA is not defined (lambda is meant to be 'beta' * 'alpha'). Compile with -DTEST_LAMBDA=<beta_times_alpha>."
#endif

static const int NUM_J = SK_N * (SK_N - 1) / 2;
static const int NUM_CONFIGS = 1024;

using coeff_t = ap_fixed<FRAC + 1, 1, AP_RND, AP_SAT>;
using delta_t = ap_fixed<FRAC + 1, 1, AP_RND, AP_SAT>;
using prob_t  = ap_ufixed<32, 1, AP_RND, AP_SAT>;


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


static ap_uint<SK_N> spin_config(int config_id) {
    /*
     * Return one of all 2^10 spin configurations.
     *
     * Since SK_N=10 and NUM_CONFIGS=1024, config_id directly labels
     * the bitstring. Bit 1 means spin +1, bit 0 means spin -1.
     */
    unsigned int raw = unsigned(config_id) & 0x3FFu;

    ap_uint<SK_N> spins = 0;
    for (int i = 0; i < SK_N; ++i) {
        spins[i] = (raw >> i) & 1u;
    }

    return spins;
}


static void fill_inputs(coeff_t h[SK_N], coeff_t J_edges[NUM_J]) {
    /*
     * Small deterministic normalized coefficients. Values are intentionally small
     * so every local Delta E / alpha stays safely inside [-1,1).
     */
    const double h_vals[SK_N] = {
        -0.0010,  0.0007, -0.0003,  0.0012, -0.0008,
         0.0004,  0.0010, -0.0006,  0.0002, -0.0009
    };

    for (int i = 0; i < SK_N; ++i) {
        h[i] = coeff_t(h_vals[i]);
    }

    for (int i = 0; i < SK_N; ++i) {
        for (int j = i + 1; j < SK_N; ++j) {
            int e = edge_index_ref(i, j);
            double v = 1.0e-4 * double(((3 * i + 5 * j) % 9) - 4);
            J_edges[e] = coeff_t(v);
        }
    }
}

static double delta_x_ref(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> spins,
    int flip_i
) {
    double field = h[flip_i].to_double();

    for (int j = 0; j < SK_N; ++j) {
        if (j != flip_i) {
            int e = edge_index_ref(flip_i, j);
            field += J_edges[e].to_double() * spin_ref(bool(spins[j]));
        }
    }

    double raw_delta = 2.0 * spin_ref(bool(spins[flip_i])) * field;
    return raw_delta > 0.0 ? raw_delta : 0.0;
}

int main() {
    coeff_t h[SK_N];
    coeff_t J_edges[NUM_J];

    fill_inputs(h, J_edges);

    const double delta_tol = std::max(128.0 * std::ldexp(1.0, -FRAC), 1e-15);
    const double prob_tol = 1.0e-4;

    int failures = 0;
    int total = 0;

    std::cout << std::setprecision(17);
    std::cout << "SK_N " << SK_N << "\n";
    std::cout << "FRAC " << FRAC << "\n";
    std::cout << "TEST_LAMBDA " << double(TEST_LAMBDA) << "\n";
    std::cout << "NUM_CONFIGS " << NUM_CONFIGS << "\n";
    std::cout << "TOTAL_TESTS " << NUM_CONFIGS * SK_N << "\n";
    std::cout << "delta_tol " << delta_tol << "\n";
    std::cout << "prob_tol " << prob_tol << "\n\n";

    std::cout << "cfg  spins  flip_i  spin  ref_x           got_x           err_x"
              << "          exact_prob     got_prob        err_prob\n";

    for (int config_id = 0; config_id < NUM_CONFIGS; ++config_id) {
        ap_uint<SK_N> spins = spin_config(config_id);

        for (int flip_i = 0; flip_i < SK_N; ++flip_i) {
            ++total;

            ap_uint<SK_N> flip_mask = 0;
            flip_mask[flip_i] = 1;

            delta_t delta_x;
            prob_t accept_prob;

            local_spin_flip_operation(h, J_edges, spins, flip_mask, &delta_x, &accept_prob);

            double ref_x = delta_x_ref(h, J_edges, spins, flip_i);
            double got_x = delta_x.to_double();
            double err_x = std::abs(got_x - ref_x);

            double exact_prob = ref_x <= 0.0 ? 1.0 : std::exp(-double(TEST_LAMBDA) * ref_x);
            double got_prob = accept_prob.to_double();
            double err_prob = std::abs(got_prob - exact_prob);

            bool ok_x = err_x <= delta_tol;
            bool ok_prob = err_prob <= prob_tol;

            if (!ok_x || !ok_prob) {
                ++failures;
            }

            /*
             * Print every failing case and a compact sample of passing cases.
             * This keeps successful runs readable while still exposing diagnostics.
             */
            bool print_case = (!ok_x || !ok_prob) || config_id < 2;

            if (print_case) {
                std::cout << std::setw(3) << config_id << " "
                        << std::setw(6) << spins.to_uint() << " "
                        << std::setw(6) << flip_i << " "
                        << std::setw(5) << spin_ref(bool(spins[flip_i])) << " "
                        << std::fixed << std::setprecision(5)
                        << std::setw(10) << ref_x << " "
                        << std::setw(10) << got_x << " "
                        << std::setw(10) << err_x << " "
                        << std::setw(10) << exact_prob << " "
                        << std::setw(10) << got_prob << " "
                        << std::setw(10) << err_prob
                        << std::defaultfloat << std::setprecision(17);

                if (!ok_x) {
                    std::cout << "  FAIL_DELTA";
                }

                if (!ok_prob) {
                    std::cout << "  FAIL_PROB";
                }

                std::cout << "\n";
            }
        }
    }

    std::cout << "\nTotal failures: " << failures << " / " << total << "\n";
    return failures == 0 ? 0 : 1;
}

