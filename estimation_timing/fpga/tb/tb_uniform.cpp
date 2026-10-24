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
                energy_t old_energy = energy_t(reference_energy(h, J, spins));
                for (int proposal = 0; proposal < 10; ++proposal) {
                    ap_uint<SK_N> proposed = proposal == 0 ? spins : proposal == 1 ? ~spins :
                        proposal == 2 ? ap_uint<SK_N>(0) : proposal == 3 ? ~ap_uint<SK_N>(0) :
                        ap_uint<SK_N>(random());
                    energy_t new_energy, delta;
                    prob_t probability;
                    uniform_move_operation(h, J, proposed, old_energy, &new_energy, &delta, &probability);
                    double expected_energy = reference_energy(h, J, proposed);
                    double expected_delta = expected_energy - old_energy.to_double();
                    require(std::abs(expected_delta) <= MAX_DELTA_ENERGY, "Delta exceeds the configured bound");
                    require(new_energy.to_double() == expected_energy, "Uniform energy is not exact on the rounded grid");
                    require(delta.to_double() == expected_delta, "Uniform energy difference is not exact on the rounded grid");
                    check_probability(expected_delta, probability, largest_error);
                    ++moves;
                }
            }
        }
        check_probability_boundaries(largest_error);
        std::cout << "PASS uniform: n=" << SK_N << ", FRAC=" << FRAC << ", integer=" << INT_BITS
                  << ", moves=" << moves << ", max acceptance error=" << largest_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL uniform n=" << SK_N << ": " << error.what() << '\n';
        return 1;
    }
}
