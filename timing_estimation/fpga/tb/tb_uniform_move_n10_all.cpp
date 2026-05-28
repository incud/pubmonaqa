/*
 * Testbench for the uniform dense-move HLS kernel.
 *
 * This testbench fixes SK_N=10 and scans all 2^10 possible old spin
 * configurations. For each old configuration, it tests a deterministic set
 * of proposed new configurations, including the same state, the complement,
 * structured states, and pseudo-random-looking bit patterns. This gives broad
 * coverage of dense non-local moves (10240) without testing all 2^20 pairs.
 *
 * For each move, the testbench calls uniform_move_operation(...). The kernel
 * receives the proposed new spin configuration and the old normalized energy
 * E_old/alpha. It computes the full dense normalized energy of the proposed
 * state,
 *
 *     E_new/alpha = -sum_i (h_i/alpha) s_i
 *                   -sum_{i<j} (J_ij/alpha) s_i s_j,
 *
 * then forms x=max((E_new-E_old)/alpha,0) and evaluates the Metropolis
 * acceptance probability.
 *
 * The testbench independently computes the same dense energy in double
 * precision and compares:
 *
 *     1. new_energy against the double-precision dense-energy reference;
 *     2. delta_x against max(E_new/alpha - E_old/alpha, 0);
 *     3. accept_prob against std::exp(-TEST_LAMBDA * delta_x),
 *        where TEST_LAMBDA = beta * alpha.
 *
 * The coefficients h and J_edges are deterministic small normalized values,
 * chosen so that normalized energies and energy differences remain safely
 * inside the intended fixed-point range. Since the uniform move recomputes
 * the full dense energy, this test also exercises all field terms, all pair
 * terms, the packed upper-triangular J_edges layout, and the log-depth adder
 * tree used for the dense energy calculation.
 *
 * The test prints all failing cases and a compact sample of passing cases.
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
#error "TEST_LAMBDA is not defined. Compile with -DTEST_LAMBDA=<beta_times_alpha>."
#endif

static const int NUM_J = SK_N * (SK_N - 1) / 2;
static const int NUM_OLD_CONFIGS = 1024;
static const int NUM_NEW_PER_OLD = 10;

using coeff_t = ap_fixed<FRAC + 1, 1, AP_RND, AP_SAT>;
using delta_t = ap_fixed<FRAC + 1, 1, AP_RND, AP_SAT>;
using prob_t  = ap_ufixed<32, 1, AP_RND, AP_SAT>;

void uniform_move_operation(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> new_spins,
    delta_t old_energy,
    delta_t* new_energy,
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
    unsigned int raw = unsigned(config_id) & 0x3FFu;

    ap_uint<SK_N> spins = 0;
    for (int i = 0; i < SK_N; ++i) {
        spins[i] = (raw >> i) & 1u;
    }

    return spins;
}

static ap_uint<SK_N> proposed_spin_config(int old_id, int proposal_id) {
    /*
     * Ten deterministic proposed states per old state.
     *
     * This keeps the test comparable in size to the local testbench:
     *   1024 old states * 10 proposals = 10240 uniform moves.
     *
     * The set includes same-state, complement, structured states, and
     * pseudo-random-looking affine states modulo 2^10.
     */
    unsigned int old_raw = unsigned(old_id) & 0x3FFu;
    unsigned int raw;

    if (proposal_id == 0) {
        raw = old_raw;                 // no-op proposal
    } else if (proposal_id == 1) {
        raw = (~old_raw) & 0x3FFu;      // complement
    } else if (proposal_id == 2) {
        raw = 0x000u;                  // all -1
    } else if (proposal_id == 3) {
        raw = 0x3FFu;                  // all +1
    } else if (proposal_id == 4) {
        raw = 0x155u;                  // alternating
    } else if (proposal_id == 5) {
        raw = 0x2AAu;                  // opposite alternating
    } else {
        raw = (73u * old_raw + 41u * unsigned(proposal_id) + 17u) & 0x3FFu;
    }

    ap_uint<SK_N> spins = 0;
    for (int i = 0; i < SK_N; ++i) {
        spins[i] = (raw >> i) & 1u;
    }

    return spins;
}

static void fill_inputs(coeff_t h[SK_N], coeff_t J_edges[NUM_J]) {
    /*
     * Small deterministic normalized coefficients. Values are intentionally small
     * so every normalized energy and uniform energy difference stays inside [-1,1).
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

static double dense_energy_ref(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> spins
) {
    double e = 0.0;

    for (int i = 0; i < SK_N; ++i) {
        e -= h[i].to_double() * spin_ref(bool(spins[i]));
    }

    for (int i = 0; i < SK_N; ++i) {
        for (int j = i + 1; j < SK_N; ++j) {
            int edge = edge_index_ref(i, j);
            e -= J_edges[edge].to_double()
               * spin_ref(bool(spins[i]))
               * spin_ref(bool(spins[j]));
        }
    }

    return e;
}

int main() {
    coeff_t h[SK_N];
    coeff_t J_edges[NUM_J];

    fill_inputs(h, J_edges);

    const double energy_tol = 1e-14;
    const double delta_tol = 1e-14;
    const double prob_tol = 1.0e-4;

    int failures = 0;
    int total = 0;

    std::cout << std::setprecision(17);
    std::cout << "SK_N " << SK_N << "\n";
    std::cout << "FRAC " << FRAC << "\n";
    std::cout << "TEST_LAMBDA " << double(TEST_LAMBDA) << "\n";
    std::cout << "NUM_OLD_CONFIGS " << NUM_OLD_CONFIGS << "\n";
    std::cout << "NUM_NEW_PER_OLD " << NUM_NEW_PER_OLD << "\n";
    std::cout << "TOTAL_TESTS " << NUM_OLD_CONFIGS * NUM_NEW_PER_OLD << "\n";
    std::cout << "energy_tol " << energy_tol << "\n";
    std::cout << "delta_tol " << delta_tol << "\n";
    std::cout << "prob_tol " << prob_tol << "\n\n";

    std::cout << "old  old_s  prop  new_s  ref_e_new     got_e_new     err_e"
              << "        ref_x       got_x       err_x       exact_p     got_p       err_p\n";

    for (int old_id = 0; old_id < NUM_OLD_CONFIGS; ++old_id) {
        ap_uint<SK_N> old_spins = spin_config(old_id);

        /*
         * The HLS kernel receives old_energy as a fixed-point input.
         * Therefore the delta reference below uses old_energy.to_double(),
         * not the infinite-precision dense_energy_ref(old_spins).
         */
        delta_t old_energy = delta_t(dense_energy_ref(h, J_edges, old_spins));
        double old_energy_input = old_energy.to_double();

        for (int proposal_id = 0; proposal_id < NUM_NEW_PER_OLD; ++proposal_id) {
            ++total;

            ap_uint<SK_N> new_spins = proposed_spin_config(old_id, proposal_id);

            delta_t new_energy;
            delta_t delta_x;
            prob_t accept_prob;

            uniform_move_operation(
                h,
                J_edges,
                new_spins,
                old_energy,
                &new_energy,
                &delta_x,
                &accept_prob
            );

            double ref_new_energy = dense_energy_ref(h, J_edges, new_spins);
            double got_new_energy = new_energy.to_double();
            double err_energy = std::abs(got_new_energy - ref_new_energy);

            double raw_delta = ref_new_energy - old_energy_input;
            double ref_x = raw_delta > 0.0 ? raw_delta : 0.0;
            double got_x = delta_x.to_double();
            double err_x = std::abs(got_x - ref_x);

            double exact_prob = ref_x <= 0.0 ? 1.0 : std::exp(-double(TEST_LAMBDA) * ref_x);
            double got_prob = accept_prob.to_double();
            double err_prob = std::abs(got_prob - exact_prob);

            bool ok_energy = err_energy <= energy_tol;
            bool ok_x = err_x <= delta_tol;
            bool ok_prob = err_prob <= prob_tol;

            if (!ok_energy || !ok_x || !ok_prob) {
                ++failures;
            }

            /*
             * Print every failing case and a compact sample of passing cases.
             */
            bool print_case = (!ok_energy || !ok_x || !ok_prob) || old_id < 2;

            if (print_case) {
                std::cout << std::setw(3) << old_id << " "
                          << std::setw(6) << old_spins.to_uint() << " "
                          << std::setw(4) << proposal_id << " "
                          << std::setw(6) << new_spins.to_uint() << " "
                          << std::fixed << std::setprecision(5)
                          << std::setw(10) << ref_new_energy << " "
                          << std::setw(10) << got_new_energy << " "
                          << std::setw(10) << err_energy << " "
                          << std::setw(10) << ref_x << " "
                          << std::setw(10) << got_x << " "
                          << std::setw(10) << err_x << " "
                          << std::setw(10) << exact_prob << " "
                          << std::setw(10) << got_prob << " "
                          << std::setw(10) << err_prob
                          << std::defaultfloat << std::setprecision(17);

                if (!ok_energy) {
                    std::cout << "  FAIL_ENERGY";
                }

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
