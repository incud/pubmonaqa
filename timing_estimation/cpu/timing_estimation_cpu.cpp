#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

#ifndef SK_N
#error "SK_N is not defined. Compile with -DSK_N=<number_of_spins>."
#endif
#ifndef N_MODELS
#error "N_MODELS is not defined. Compile with -DN_MODELS=<number_of_models>."
#endif
#ifndef N_STEPS
#error "N_STEPS is not defined. Compile with -DN_STEPS=<number_of_steps>."
#endif
#ifndef SEED
#error "SEED is not defined. Compile with -DSEED=<seed>."
#endif
#ifndef BETA
#define BETA 1.0f
#endif

/** Dense SK model with float coefficients.
 *  J is stored as a full N x N row-major matrix for cheap indexing in kernels.
 */
template <std::size_t N>
struct IsingModel {
    alignas(64) std::array<float, N> h{};
    alignas(64) std::array<float, N * N> J{};
};

/** Unpacked spin state, with entries in {-1, +1}.
 *  This uses more memory than a bitstring but avoids bit extraction in dense loops.
 */
template <std::size_t N>
using IsingState = std::array<int8_t, N>;

/** Precomputed random numbers used to keep RNG cost outside timed regions. */
struct RandPool {
    std::vector<uint64_t> ri;
    std::vector<float> ru;
    std::size_t ii = 0;
    std::size_t iu = 0;

    RandPool(std::size_t n_int, std::size_t n_float, uint64_t seed) : ri(n_int), ru(n_float) {
        // Fill both pools once before benchmarking.
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<uint64_t> di;
        std::uniform_real_distribution<float> du(0.0f, 1.0f);
        for (auto& x : ri) {
            x = di(gen);
        }
        for (auto& x : ru) {
            x = du(gen);
        }
    }

    /** Reuse the same random sequence for the next timed kernel. */
    void reset() {
        ii = 0;
        iu = 0;
    }

    /** Return the next precomputed 64-bit integer. */
    uint64_t randint() {
        return ri[ii++];
    }

    /** Return the next precomputed uniform float in [0,1). */
    float uniform() {
        return ru[iu++];
    }
};

/** Draw a random spin configuration from the precomputed integer pool. */
template <std::size_t N>
void random_state(IsingState<N>& state, RandPool& rng) {
    // One 64-bit word generates up to 64 unpacked spins.
    for (std::size_t base = 0; base < N; base += 64) {
        uint64_t w = rng.randint();
        std::size_t end = base + 64 < N ? base + 64 : N;
        for (std::size_t i = base; i < end; ++i) {
            state[i] = ((w >> (i - base)) & 1ULL) ? -1 : 1;
        }
    }
}

/** Initialize one spherical SK instance.
 *  Draw all independent coefficients from N(0,1), then rescale so that
 *  sum_i h_i^2 + sum_{i<j} J_ij^2 = N.
 */
template <std::size_t N>
void init_model(IsingModel<N>& model, std::mt19937_64& gen) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    float ssq = 0.0f;

    // Local fields.
    for (std::size_t i = 0; i < N; ++i) {
        model.h[i] = normal(gen);
        ssq += model.h[i] * model.h[i];
    }

    // Symmetric dense coupling matrix, with zero diagonal by default.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            float v = normal(gen);
            model.J[i * N + j] = v;
            model.J[j * N + i] = v;
            ssq += v * v;
        }
    }

    // Spherical normalization of the coefficient vector.
    float scale = std::sqrt(float(N) / ssq);
    for (float& v : model.h) {
        v *= scale;
    }
    for (float& v : model.J) {
        v *= scale;
    }
}

/** Compute the dense Ising energy E(s).
 *  This is O(N^2) and is used by the uniform full-state proposal.
 */
template <std::size_t N>
float energy(const IsingModel<N>& model, const IsingState<N>& state) {
    float e = 0.0f;

    // Field contribution: -sum_i h_i s_i.
    for (std::size_t i = 0; i < N; ++i) {
        e -= model.h[i] * state[i];
    }

    // Coupling contribution: -sum_{i<j} J_ij s_i s_j.
    for (std::size_t i = 0; i < N; ++i) {
        int si = state[i];
        for (std::size_t j = i + 1; j < N; ++j) {
            e -= model.J[i * N + j] * si * state[j];
        }
    }
    return e;
}

/** Compute the O(N) single-spin-flip energy difference.
 *  Delta_i E = 2 s_i (h_i + sum_j J_ij s_j).
 */
template <std::size_t N>
float delta_local(const IsingModel<N>& model, const IsingState<N>& state, std::size_t i) {
    // Unpacked spins avoid bit extraction inside this dense row dot product.
    float field = model.h[i];
    for (std::size_t j = 0; j < N; ++j) {
        field += model.J[i * N + j] * state[j];
    }
    return 2.0f * state[i] * field;
}

/** Perform one local Metropolis step with a single-spin proposal. */
template <std::size_t N>
void do_local_step(const IsingModel<N>& model, IsingState<N>& state, float& e, RandPool& rng) {
    // Pick the proposed spin and compute the exact O(N) delta energy.
    std::size_t i = rng.randint() % N;
    float dE = delta_local(model, state, i);

    // Downhill moves are accepted; uphill moves use exp(-beta dE).
    if (dE <= 0.0f || rng.uniform() < std::exp(-static_cast<float>(BETA) * dE)) {
        state[i] = -state[i];
        e += dE;
    }
}

/** Perform one Metropolis step with a uniform full-state proposal. */
template <std::size_t N>
void do_uniform_step(const IsingModel<N>& model, IsingState<N>& state, float& e, RandPool& rng) {
    // Full random proposal; the new dense energy evaluation dominates this kernel.
    IsingState<N> trial;
    random_state(trial, rng);

    float e_new = energy(model, trial);
    float dE = e_new - e;

    // Accept/reject using the usual Metropolis rule.
    if (dE <= 0.0f || rng.uniform() < std::exp(-static_cast<float>(BETA) * dE)) {
        state = trial;
        e = e_new;
    }
}

/** Reset all chains to the all-plus state and recompute their energies. */
template <std::size_t N>
void reset_states(
    std::vector<IsingState<N>>& states,
    std::vector<float>& energies,
    const std::vector<IsingModel<N>>& models
) {
    for (std::size_t m = 0; m < states.size(); ++m) {
        states[m].fill(1);
        energies[m] = energy(models[m], states[m]);
    }
}

int main() {
    // Enough random integers for the worst case: one full-state proposal per operation.
    constexpr std::size_t words = (SK_N + 63) / 64;
    const std::size_t ops = std::size_t(N_STEPS) * std::size_t(N_MODELS);

    RandPool rng(ops * words, ops, SEED);
    std::mt19937_64 gen(SEED);

    // Generate all models outside the timed region.
    std::vector<IsingModel<SK_N>> models(N_MODELS);
    for (auto& model : models) {
        init_model(model, gen);
    }

    std::vector<IsingState<SK_N>> states(N_MODELS);
    std::vector<float> energies(N_MODELS);

    // Benchmark local single-spin proposals.
    reset_states(states, energies, models);
    rng.reset();
    auto t0 = Clock::now();
    for (std::size_t t = 0; t < N_STEPS; ++t) {
        for (std::size_t m = 0; m < N_MODELS; ++m) {
            do_local_step(models[m], states[m], energies[m], rng);
        }
    }
    auto t1 = Clock::now();

    // Benchmark uniform full-state proposals.
    reset_states(states, energies, models);
    rng.reset();
    auto t2 = Clock::now();
    for (std::size_t t = 0; t < N_STEPS; ++t) {
        for (std::size_t m = 0; m < N_MODELS; ++m) {
            do_uniform_step(models[m], states[m], energies[m], rng);
        }
    }
    auto t3 = Clock::now();

    std::cout << "SK_N " << SK_N << "\n";
    std::cout << "N_MODELS " << N_MODELS << "\n";
    std::cout << "N_STEPS " << N_STEPS << "\n";
    std::cout << "local_seconds  " << std::chrono::duration<double>(t1 - t0).count() << "\n";
    std::cout << "uniform_seconds " << std::chrono::duration<double>(t3 - t2).count() << "\n";
}
