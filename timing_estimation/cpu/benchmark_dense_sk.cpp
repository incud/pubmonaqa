#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef SK_N
#error "SK_N is not defined. Compile with -DSK_N=<number_of_spins>, e.g. g++ -O3 -march=native -DNDEBUG -std=c++20 -DSK_N=50 benchmark_dense_sk_fixedN_bitset_v5.cpp -o dense_sk_bench_N50"
#endif

static_assert(SK_N > 0, "SK_N must be positive.");

using Clock = std::chrono::steady_clock;

struct Options {
    int instances = 100;
    std::uint64_t steps = 1'000'000ULL;
    double beta = 1.0;
    std::uint64_t seed = 123456789ULL;
    std::string out = "dense_sk_single_move_N" + std::to_string(SK_N) + ".csv";
};

/** Print command-line usage and terminate. */
[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n\n"
        << "  --instances INT    Default: 100\n"
        << "  --steps INT        Default: 1000000\n"
        << "  --beta FLOAT       Default: 1.0\n"
        << "  --seed INT         Default: 123456789\n"
        << "  --out PATH         Default: dense_sk_single_move_N" << SK_N << ".csv\n"
        << "  --help\n\n"
        << "The program always benchmarks both local1 and uniform moves for compile-time SK_N=" << SK_N << ".\n"
        << "Example compile command:\n"
        << "  g++ -O3 -march=native -DNDEBUG -std=c++20 -DSK_N=" << SK_N << ' ' << argv0
        << " -o dense_sk_bench_N" << SK_N << "\n";
    std::exit(1);
}

/** Parse command-line options. Move type is not an option: both moves are always measured. */
Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto next = [&](std::string_view name) -> const char* {
            if (++i >= argc) {
                std::cerr << "Missing value for " << name << '\n';
                usage(argv[0]);
            }
            return argv[i];
        };
        if (a == "--help" || a == "-h") usage(argv[0]);
        else if (a == "--instances") o.instances = std::stoi(next(a));
        else if (a == "--steps") o.steps = std::stoull(next(a));
        else if (a == "--beta") o.beta = std::stod(next(a));
        else if (a == "--seed") o.seed = std::stoull(next(a));
        else if (a == "--out") o.out = next(a);
        else {
            std::cerr << "Unknown option: " << a << '\n';
            usage(argv[0]);
        }
    }
    if (o.instances <= 0) throw std::invalid_argument("instances must be positive");
    if (o.steps == 0) throw std::invalid_argument("steps must be positive");
    if (!std::isfinite(o.beta) || o.beta < 0.0) throw std::invalid_argument("beta must be finite and nonnegative");
    return o;
}

/** Return a uniform double in [0,1) from the upper 53 random bits. */
inline double uniform01(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
}

/** Dense SK instance with symmetric zero-diagonal J stored row-major as an N x N array. */
template <std::size_t N>
struct DenseSK {
    double alpha = 1.0;
    alignas(64) std::array<double, N> h{};
    alignas(64) std::array<double, N * N> J{};
};

/** Convert one bit to a spin: bit 1 means +1, bit 0 means -1. */
template <std::size_t N>
inline double spin(const std::bitset<N>& s, std::size_t i) {
    return s.test(i) ? 1.0 : -1.0;
}

/** Fill a bitset with random spins using the single global RNG stream. */
template <std::size_t N>
void randomize_spins(std::bitset<N>& s, std::mt19937_64& rng) {
    for (std::size_t i = 0; i < N; ++i) {
        // Consume one bit per spin; this is faster than drawing a distribution per site.
        s.set(i, static_cast<bool>(rng() & 1ULL));
    }
}

/**
 * Generate one dense SK instance from the single RNG stream.
 *
 * Coefficients are sampled from N(0,1), J is symmetric with zero diagonal, and then
 * h,J are rescaled by alpha = sqrt(N / (sum_i h_i^2 + sum_{i<j} J_ij^2)).
 */
template <std::size_t N>
DenseSK<N> generate_instance(std::mt19937_64& rng) {
    DenseSK<N> m;
    std::normal_distribution<double> normal(0.0, 1.0);

    double norm2 = 0.0;
    for (double& hi : m.h) {
        hi = normal(rng);
        norm2 += hi * hi;
    }
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            const double Jij = normal(rng);
            m.J[i * N + j] = m.J[j * N + i] = Jij;
            norm2 += Jij * Jij;  // Count each independent coupling once.
        }
    }

    m.alpha = std::sqrt(static_cast<double>(N) / norm2);
    for (double& hi : m.h) hi *= m.alpha;
    for (double& Jij : m.J) Jij *= m.alpha;
    return m;
}

/** Compute E(s) = -sum_i h_i s_i - sum_{i<j} J_ij s_i s_j. */
template <std::size_t N>
double energy(const DenseSK<N>& m, const std::bitset<N>& s) {
    double e = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        e -= m.h[i] * spin(s, i);
    }
    for (std::size_t i = 0; i < N; ++i) {
        const double si = spin(s, i);
        const double* row = m.J.data() + i * N;
        for (std::size_t j = i + 1; j < N; ++j) {
            // Only upper-triangular pairs are summed, matching sum_{i<j} J_ij s_i s_j.
            e -= si * row[j] * spin(s, j);
        }
    }
    return e;
}

/** Return Delta E for flipping one spin, using only the O(N) terms touching that spin. */
template <std::size_t N>
double delta_energy_single_flip(const DenseSK<N>& m, const std::bitset<N>& s, std::size_t i) {
    const double* row = m.J.data() + i * N;
    double local_field = m.h[i];
    for (std::size_t j = 0; j < N; ++j) {
        // J_ii is zero, so including j=i is harmless and avoids a branch in the hot loop.
        local_field += row[j] * spin(s, j);
    }
    return 2.0 * spin(s, i) * local_field;
}

/** Return Delta E for replacing the current state by a uniform proposal; this is O(N^2). */
template <std::size_t N>
inline double delta_energy_uniform_proposal(const DenseSK<N>& m, double current_energy, const std::bitset<N>& proposal) {
    return energy(m, proposal) - current_energy;
}

/** Metropolis accept/reject test for an energy increment at inverse temperature beta. */
inline bool metropolis_accept(double delta_e, double beta, std::mt19937_64& rng) {
    // Negative-energy moves are always accepted; otherwise use native double exp.
    return delta_e <= 0.0 || beta == 0.0 || uniform01(rng) < std::exp(-beta * delta_e);
}

struct Result {
    double seconds = 0.0;
    double seconds_per_move = 0.0;
    std::uint64_t accepted = 0;
    double final_energy = 0.0;
};

/** Time local single-spin Metropolis proposals with the O(N) Delta-E formula. */
template <std::size_t N>
Result bench_local1(const DenseSK<N>& m, const std::bitset<N>& initial, std::uint64_t steps, double beta, std::mt19937_64& rng) {
    std::bitset<N> s = initial;  // Work on a copy so the uniform benchmark can use the same initial state.
    double e = energy(m, s);     // Current energy is maintained incrementally after accepted moves.
    std::uint64_t accepted = 0;

    const auto start = Clock::now();
    for (std::uint64_t t = 0; t < steps; ++t) {
        const std::size_t i = static_cast<std::size_t>(rng() % N);
        const double de = delta_energy_single_flip(m, s, i);
        if (metropolis_accept(de, beta, rng)) {
            s.flip(i);
            e += de;  // Same update rule as the uniform move: new_energy = old_energy + Delta E.
            ++accepted;
        }
    }
    const auto stop = Clock::now();

    const double sec = std::chrono::duration<double>(stop - start).count();
    return {sec, sec / static_cast<double>(steps), accepted, e};
}

/** Time uniform global Metropolis proposals with the O(N^2) Delta-E formula. */
template <std::size_t N>
Result bench_uniform(const DenseSK<N>& m, const std::bitset<N>& initial, std::uint64_t steps, double beta, std::mt19937_64& rng) {
    std::bitset<N> s = initial;
    std::bitset<N> proposal;
    double e = energy(m, s);
    std::uint64_t accepted = 0;

    const auto start = Clock::now();
    for (std::uint64_t t = 0; t < steps; ++t) {
        randomize_spins(proposal, rng);                  // Propose y uniformly from {+-1}^N.
        const double de = delta_energy_uniform_proposal(m, e, proposal);
        if (metropolis_accept(de, beta, rng)) {
            s = proposal;
            e += de;                                    // new_energy = old_energy + Delta E.
            ++accepted;
        }
    }
    const auto stop = Clock::now();

    const double sec = std::chrono::duration<double>(stop - start).count();
    return {sec, sec / static_cast<double>(steps), accepted, e};
}

/** Write the CSV header. */
void write_header(std::ostream& out) {
    out << "n,instance,move,steps,beta,seconds,seconds_per_move,ns_per_move,accepted,acceptance_rate,final_energy,alpha\n";
}

/** Write one benchmark result as a CSV row. */
void write_row(std::ostream& out, int instance, std::string_view move, const Options& o, double alpha, const Result& r) {
    out << SK_N << ',' << instance << ',' << move << ',' << o.steps << ','
        << std::setprecision(17) << o.beta << ',' << r.seconds << ',' << r.seconds_per_move << ','
        << 1e9 * r.seconds_per_move << ',' << r.accepted << ','
        << static_cast<double>(r.accepted) / static_cast<double>(o.steps) << ','
        << r.final_energy << ',' << alpha << '\n';
}

/** Generate instances, benchmark both move types, and write timing rows. */
int main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        std::ofstream out(o.out);
        if (!out) throw std::runtime_error("could not open output file: " + o.out);
        write_header(out);

        std::mt19937_64 rng(o.seed);  // Single global RNG stream: no seed mixing or per-instance RNGs.
        std::cerr << "SK_N=" << SK_N << ", instances=" << o.instances << ", steps=" << o.steps << '\n';

        for (int id = 0; id < o.instances; ++id) {
            const DenseSK<SK_N> m = generate_instance<SK_N>(rng);

            std::bitset<SK_N> initial;
            randomize_spins(initial, rng);

            write_row(out, id, "local1", o, m.alpha, bench_local1(m, initial, o.steps, o.beta, rng));
            write_row(out, id, "uniform", o, m.alpha, bench_uniform(m, initial, o.steps, o.beta, rng));

            if ((id + 1) % 10 == 0 || id + 1 == o.instances) {
                std::cerr << "  completed instance " << (id + 1) << '/' << o.instances << '\n';
                out.flush();
            }
        }

        std::cerr << "wrote " << o.out << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
