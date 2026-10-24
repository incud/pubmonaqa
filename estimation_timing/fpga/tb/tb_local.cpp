#include "reference.hpp"

int main() {
    try {
        constexpr int CASES = 1 << (SK_N <= 10 ? SK_N : 5);
        constexpr int MODELS = SK_N <= 10 ? 2 : 1;
        double largest_error = 0;
        long moves = 0;
        std::mt19937_64 random(42);
        for (int model = 0; model < MODELS; ++model) {
            coeff_t h[SK_N], J[NUM_J];
            make_model(h, J, 42 + model);
            for (int state = 0; state < CASES; ++state) {
                ap_uint<SK_N> spins = configuration(state, random);
                double old_energy = reference_energy(h, J, spins);
                for (int i = 0; i < SK_N; ++i) {
                    ap_uint<SK_N> mask = 0;
                    mask[i] = 1;
                    energy_t delta;
                    prob_t probability;
                    local_spin_flip_operation(h, J, spins, mask, &delta, &probability);
                    double expected = reference_energy(h, J, spins ^ mask) - old_energy;
                    require(std::abs(expected) <= MAX_DELTA_ENERGY, "Delta exceeds the configured bound");
                    require(delta.to_double() == expected, "Local energy difference is not exact on the rounded grid");
                    check_probability(expected, probability, largest_error);
                    ++moves;
                }
            }
        }
        check_probability_boundaries(largest_error);
        std::cout << "PASS local: n=" << SK_N << ", FRAC=" << FRAC << ", integer=" << INT_BITS
                  << ", moves=" << moves << ", max acceptance error=" << largest_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL local n=" << SK_N << ": " << error.what() << '\n';
        return 1;
    }
}
