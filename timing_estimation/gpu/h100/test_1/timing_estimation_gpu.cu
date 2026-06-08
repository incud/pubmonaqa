#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#ifndef SK_N
#error "SK_N is not defined. Compile with -DSK_N=<number_of_spins>."
#endif
#ifndef N_MODELS
#error "N_MODELS is not defined. Compile with -DN_MODELS=<number_of_models>."
#endif
#ifndef CHAINS_PER_MODEL
#error "CHAINS_PER_MODEL is not defined. Compile with -DCHAINS_PER_MODEL=<chains_per_model>."
#endif
#ifndef N_STEPS
#error "N_STEPS is not defined. Compile with -DN_STEPS=<number_of_steps>."
#endif
#ifndef SEED
#error "SEED is not defined. Compile with -DSEED=<seed>."
#endif
#ifndef BETA
#error "BETA is not defined. Compile with -DBETA=<inverse_temperature>."
#endif

// BLOCK_THREADS is the number of CUDA threads assigned to one MCMC chain.
// I.e. one CUDA block = one independent MCMC chain; BLOCK_THREADS = worker threads inside that block.
#ifndef BLOCK_THREADS
#define BLOCK_THREADS 128
#endif

// To avoid benchmarking host RNG, we create a small random pool before timing.
// The pool is reused cyclically during the kernels; this is for timing, not production MCMC.
#ifndef RNG_POOL_SIZE
#define RNG_POOL_SIZE 1048576
#endif

// Total number of independent chains executed by the GPU.
#define N_CHAINS (N_MODELS * CHAINS_PER_MODEL)

// This macro exits the program whenever a CUDA runtime call or checked kernel launch fails.
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n"; std::exit(1); }} while (0)


// DeviceArray<T> is a wrapper for GPU memory: keeps cudaMalloc/cudaFree ownership in one place,
// prevents leaks, and keeps the benchmark logic free of raw allocation boilerplate.
template <class T>
struct DeviceArray {
    // Raw pointer to the GPU allocation.
    T* ptr = nullptr;

    // Number of elements allocated on the GPU.
    std::size_t n = 0;

    // Empty wrapper, used before an allocation is needed.
    DeviceArray() = default;

    // Allocate n_ elements of type T on the GPU.
    explicit DeviceArray(std::size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }

    // Free the GPU allocation when the wrapper goes out of scope.
    ~DeviceArray() { if (ptr) { cudaFree(ptr); } }

    // Disable copying, because two wrappers must not own the same GPU pointer.
    DeviceArray(const DeviceArray&) = delete;

    // Disable copy assignment for the same ownership reason.
    DeviceArray& operator=(const DeviceArray&) = delete;

    // Mutable access to the raw GPU pointer, used when launching kernels.
    T* data() { return ptr; }

    // Const access to the raw GPU pointer.
    const T* data() const { return ptr; }

    // Copy a host vector into this GPU allocation.
    void copy_from(const std::vector<T>& v) { CUDA_CHECK(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
};


// We now create a small CUDA event timer, so kernel timing does not clutter main().
struct GpuTimer {
    cudaEvent_t a;
    cudaEvent_t b;

    // Allocate two CUDA events used as start/stop markers.
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
    }

    // Destroy CUDA timing events.
    ~GpuTimer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }

    // Mark the beginning of a timed GPU region.
    void start() {
        CUDA_CHECK(cudaEventRecord(a));
    }

    // Mark the end of a timed GPU region and return elapsed seconds.
    float stop_seconds() {
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms / 1000.0f;
    }
};


// We now create a wrapper for the random number pool. This is composed of a CPU-side
// structure named 'RandomPool' that owns CPU vectors and GPU allocations, so it is a host object;
// and 'RandomPoolView' is a tiny GPU-compatible object containing only device pointers and __device__ methods.

struct RandomPoolView {
    // Device-side precomputed integer random numbers.
    const uint64_t* rint;

    // Device-side precomputed uniform floats in [0,1).
    const float* rfloat;

    // Return a pseudo-random integer from the reused pool.
    __device__ uint64_t randint(uint64_t chain, uint64_t step, uint64_t tag) const {
        return rint[(chain + 1315423911ULL * step + 2654435761ULL * tag) % RNG_POOL_SIZE];
    }

    // Return a pseudo-random uniform float from the reused pool.
    __device__ float uniform(uint64_t chain, uint64_t step, uint64_t tag) const {
        return rfloat[(chain + 11400714819323198485ULL * step + 14029467366897019727ULL * tag) % RNG_POOL_SIZE];
    }

    // Return one random spin in {-1,+1}.
    __device__ int8_t spin(uint64_t chain, uint64_t step, int i) const {
        uint64_t w = randint(chain, step, 1000 + i / 64);
        return ((w >> (i & 63)) & 1ULL) ? -1 : 1;
    }

};


struct RandomPool {
    // Host-side precomputed integer random numbers.
    std::vector<uint64_t> h_int;

    // Host-side precomputed uniform floats in [0,1).
    std::vector<float> h_float;

    // Device-side integer random pool.
    DeviceArray<uint64_t> d_int;

    // Device-side uniform random pool.
    DeviceArray<float> d_float;

    // Generate host pools, allocate device pools, and copy them to the GPU.
    RandomPool(std::size_t n, uint64_t seed)
        : h_int(n), h_float(n), d_int(n), d_float(n) {
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<uint64_t> int_dist;
        std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);
        for (auto& x : h_int) {
            x = int_dist(gen);
        }
        for (auto& x : h_float) {
            x = float_dist(gen);
        }
        d_int.copy_from(h_int);
        d_float.copy_from(h_float);
    }

    // Give the GPU-compatible view object to the kernels.
    RandomPoolView view() const {
        return RandomPoolView{d_int.data(), d_float.data()};
    }
};


// We now create a model bank. The host object owns all SK instances, while the view
// gives kernels cheap access to the h and J arrays of the model assigned to a chain.

struct ModelBankView {
    // Device-side fields h, stored as [N_MODELS, SK_N].
    const float* h;

    // Device-side dense couplings J, stored as [N_MODELS, SK_N, SK_N].
    const float* J;

    // Return the h array for one model.
    __device__ const float* h_model(int model) const {
        return h + model * SK_N;
    }

    // Return the J matrix for one model.
    __device__ const float* J_model(int model) const {
        return J + model * SK_N * SK_N;
    }
};



struct ModelBank {
    // Host-side fields for all models.
    std::vector<float> h_host;

    // Host-side dense coupling matrices for all models.
    std::vector<float> J_host;

    // Device-side fields for all models.
    DeviceArray<float> h_dev;

    // Device-side dense coupling matrices for all models.
    DeviceArray<float> J_dev;

    // Generate spherical SK models and copy them to the GPU.
    ModelBank(uint64_t seed)
        : h_host(size_t(N_MODELS) * SK_N),
          J_host(size_t(N_MODELS) * SK_N * SK_N),
          h_dev(h_host.size()),
          J_dev(J_host.size()) {
        init_models(seed);
        h_dev.copy_from(h_host);
        J_dev.copy_from(J_host);
    }

    // Host-side dense energy for one model and one spin state.
    float energy(int model, const int8_t* s) const {
        const float* h = h_host.data() + size_t(model) * SK_N;
        const float* J = J_host.data() + size_t(model) * SK_N * SK_N;
        float e = 0.0f;

        for (int i = 0; i < SK_N; ++i) {
            e -= h[i] * float(s[i]);
        }

        for (int i = 0; i < SK_N; ++i) {
            int si = int(s[i]);
            for (int j = i + 1; j < SK_N; ++j) {
                e -= J[i * SK_N + j] * si * int(s[j]);
            }
        }

        return e;
    }


    // Fill h_host and J_host with spherical SK instances.
    void init_models(uint64_t seed) {
        std::mt19937_64 gen(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        for (int m = 0; m < N_MODELS; ++m) {
            float ssq = 0.0f;
            float* h = h_host.data() + size_t(m) * SK_N;
            float* J = J_host.data() + size_t(m) * SK_N * SK_N;

            for (int i = 0; i < SK_N; ++i) {
                h[i] = normal(gen);
                ssq += h[i] * h[i];
            }

            for (int i = 0; i < SK_N; ++i) {
                for (int j = i + 1; j < SK_N; ++j) {
                    float x = normal(gen);
                    J[i * SK_N + j] = x;
                    J[j * SK_N + i] = x;
                    ssq += x * x;
                }
            }

            float scale = std::sqrt(float(SK_N) / ssq);

            for (int i = 0; i < SK_N; ++i) {
                h[i] *= scale;
            }

            for (int i = 0; i < SK_N * SK_N; ++i) {
                J[i] *= scale;
            }
        }
    }

    // Give the GPU-compatible view object to the kernels.
    ModelBankView view() const {
        return ModelBankView{h_dev.data(), J_dev.data()};
    }
};


// We now create the chain state. It owns the device spin states and energies, while
// ChainStateView gives kernels access to the state and energy of a specific chain.

struct ChainStateView {
    // Device-side spin states, stored as [N_CHAINS, SK_N].
    int8_t* s;

    // Device-side current energies, one per chain.
    float* E;

    // Return the spin array for one chain.
    __device__ int8_t* state(int chain) const {
        return s + chain * SK_N;
    }

    // Return a reference to the energy of one chain.
    __device__ float& energy(int chain) const {
        return E[chain];
    }

    // Flip one spin in one chain.
    __device__ void flip(int chain, int i) const {
        int8_t* sc = state(chain);
        sc[i] = -sc[i];
    }

    // Reset one chain to the all-plus state using all threads of one CUDA block.
    __device__ void reset_state(int chain) const {
        int tid = threadIdx.x;
        int8_t* sc = state(chain);

        for (int i = tid; i < SK_N; i += blockDim.x) {
            sc[i] = 1;
        }
    }
};


struct ChainState {
    // Host-side initial all-plus energy, one value per chain.
    std::vector<float> E0_host;

    // Device-side spin states.
    DeviceArray<int8_t> s_dev;

    // Device-side current energies.
    DeviceArray<float> E_dev;

    // Allocate chain states and compute the initial all-plus energy for each chain.
    explicit ChainState(const ModelBank& models)
        : E0_host(N_CHAINS),
          s_dev(size_t(N_CHAINS) * SK_N),
          E_dev(N_CHAINS) {
        init_initial_energies(models);
    }

    // Compute the initial all-plus energy for every chain.
    void init_initial_energies(const ModelBank& models) {
        std::vector<int8_t> all_plus(SK_N, 1);

        for (int m = 0; m < N_MODELS; ++m) {
            float e = models.energy(m, all_plus.data());

            for (int k = 0; k < CHAINS_PER_MODEL; ++k) {
                E0_host[size_t(m) * CHAINS_PER_MODEL + k] = e;
            }
        }
    }

    // Reset device energies to the all-plus initial values.
    void reset_energy() {
        E_dev.copy_from(E0_host);
    }

    // Give the GPU-compatible view object to the kernels.
    ChainStateView view() {
        return ChainStateView{s_dev.data(), E_dev.data()};
    }
};


// We now define the per-block object owning the shared memory used by one CUDA block:
// * a reduction buffer used by block-level sums in the local and uniform energy calculations;
// * a trial spin state buffer used by the uniform proposal before it is accepted.
// The methods below are collective block operations, so every thread in the block
// must call them whenever they are executed.

struct IsingEnergyBuffer {

    // Shared reduction buffer used by local and uniform energy calculations.
    float red[BLOCK_THREADS];

    // Shared trial state used only by the uniform proposal.
    int8_t trial[SK_N];

    // Sum one contribution per thread inside the CUDA block.
    __device__ float block_sum(float x) {
        int tid = threadIdx.x;

        // First, each thread stores its partial contribution in the shared reduction buffer.
        red[tid] = x;
        __syncthreads();

        // Then a binary-tree reduction sums the blockDim.x partial values in log2(blockDim.x) steps.
        // This assumes BLOCK_THREADS is a power of two.
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off) {
                red[tid] += red[tid + off];
            }
            __syncthreads();
        }

        // The summed value is now stored in the first entry of the block-local buffer.
        return red[0];
    }

    // Compute Delta_i E = 2 s_i (h_i + sum_j J_ij s_j) for a local single-spin flip.
    // This requires one dense row dot product, hence O(N) arithmetic work per proposal.
    // The work is split across BLOCK_THREADS threads, so each thread handles roughly
    // O(N / BLOCK_THREADS) terms before the shared reduction combines the partial sums.
    __device__ float local_delta_energy(const float* h, const float* J, const int8_t* s, int i) {
        int tid = threadIdx.x;
        float part = tid == 0 ? h[i] : 0.0f;

        for (int j = tid; j < SK_N; j += blockDim.x) {
            part += J[i * SK_N + j] * float(s[j]);
        }

        float field = block_sum(part);
        return 2.0f * float(s[i]) * field;
    }

    // Generate the internal trial state using the precomputed random pool.
    // The spin indices are split across the block, so each thread initializes roughly
    // O(N / BLOCK_THREADS) spins.
    __device__ void assign_random_state_to_trial(RandomPoolView rng, uint64_t chain, uint64_t step) {
        int tid = threadIdx.x;

        for (int i = tid; i < SK_N; i += blockDim.x) {
            trial[i] = rng.spin(chain, step, i);
        }

        __syncthreads();
    }

    // Compute the dense energy of the internal trial state generated by the uniform move.
    // This includes all field terms and all pair couplings, so the total arithmetic work is
    // O(N^2), split across the block threads before the shared reduction.
    __device__ float get_energy_trial_state(const float* h, const float* J) {
        int tid = threadIdx.x;
        float part = 0.0f;

        for (int i = tid; i < SK_N; i += blockDim.x) {
            int si = int(trial[i]);
            part -= h[i] * si;

            for (int j = i + 1; j < SK_N; ++j) {
                part -= J[i * SK_N + j] * si * int(trial[j]);
            }
        }

        return block_sum(part);
    }

    // Copy the internal trial state into the persistent chain state.
    __device__ void accept_trial_state(int8_t* s) {
        int tid = threadIdx.x;

        for (int i = tid; i < SK_N; i += blockDim.x) {
            s[i] = trial[i];
        }
    }
};


// Metropolis accept/reject rule. This does not use shared memory, so it stays
// outside IsingEnergyBuffer.
__device__ bool metropolis_accept(float dE, RandomPoolView rng, uint64_t chain, uint64_t step, uint64_t tag) {
    // method '__expf' needs to compile with flag --use_fast_math
    return (dE <= 0.0f) || (rng.uniform(chain, step, tag) < __expf(-float(BETA) * dE));
}


// We now define the __global__ kernels. These are the functions launched from the CPU.
// 1. reset_states: resets every chain to the all-plus spin state.
// 2. local_kernel: performs local single-spin Metropolis steps.
// 3. uniform_kernel: performs uniform full-state Metropolis steps.

// Reset all chains to the all-plus state on the GPU.
__global__ void reset_states(ChainStateView chains) {
    int chain = blockIdx.x;
    chains.reset_state(chain);
}


// Local proposal kernel. One CUDA block is one chain.
__global__ void local_kernel(ModelBankView models, ChainStateView chains, RandomPoolView rng) {
    int chain = blockIdx.x;                     // one CUDA block is assigned to one MCMC chain.
    int tid = threadIdx.x;
    int model = chain / CHAINS_PER_MODEL;       // each group of CHAINS_PER_MODEL chains reuses one SK model.

    const float* h = models.h_model(model);     // fields of the model assigned to this chain.
    const float* J = models.J_model(model);     // dense coupling matrix of the assigned model.
    int8_t* s = chains.state(chain);            // persistent spin state of this chain.

    __shared__ IsingEnergyBuffer energy_helper; // per-chain scratch space shared by the block threads.
    __shared__ int i;                           // proposed spin index chosen by thread 0 and read by all threads.
    __shared__ int accept;                      // Metropolis decision computed by thread 0 and used to control the block.
    __shared__ float dE;                        // proposed energy difference, stored for the accepted energy update.

    // All the steps here must be performed sequentially, 
    // albeit many local_kernel are executed in parallel. 
    for (uint64_t step = 0; step < N_STEPS; ++step) {

        // Only thread 0 chooses which spin to flip; all other threads use this index
        // to cooperatively compute the corresponding local energy difference.
        if (tid == 0) {
            i = int(rng.randint(chain, step, 0) % SK_N);
        }

        // Sync here to make sure all threads see variable 'i' before computing the local energy.
        __syncthreads();

        // All threads must call local_delta_energy because it performs a block-level
        // reduction and therefore contains synchronization barriers.
        float dE_tmp = energy_helper.local_delta_energy(h, J, s, i);

        // Only the first thread computes the acceptance probability 
        // and updates the shared state (no data races)
        if (tid == 0) {
            dE = dE_tmp;
            accept = metropolis_accept(dE, rng, chain, step, 0);

            if (accept) {
                chains.flip(chain, i);
                chains.energy(chain) += dE;
            }
        }

        // Sync here to make sure all threads see the new state before the next MCMC step.
        __syncthreads();
    }
}


// Uniform proposal kernel. One CUDA block is one chain.
__global__ void uniform_kernel(ModelBankView models, ChainStateView chains, RandomPoolView rng) {
    int chain = blockIdx.x;
    int tid = threadIdx.x;
    int model = chain / CHAINS_PER_MODEL;

    const float* h = models.h_model(model);
    const float* J = models.J_model(model);
    int8_t* s = chains.state(chain);

    __shared__ IsingEnergyBuffer energy_helper;         // per-chain scratch space for the trial state and reductions.
    __shared__ int accept;                              // Metropolis decision computed by thread 0 and read by all threads.
    __shared__ float e_new;                             // trial-state energy computed by the block and committed if accepted.

    for (uint64_t step = 0; step < N_STEPS; ++step) {

        // All threads must call assign_random_state_to_trial because they jointly
        // initialize the trial state and the method contains a synchronization barrier.
        energy_helper.assign_random_state_to_trial(rng, chain, step);

        // All threads must call get_energy_trial_state because it computes a block-wide
        // dense energy reduction and therefore contains synchronization barriers.
        float e_tmp = energy_helper.get_energy_trial_state(h, J);

        // Only thread 0 computes the acceptance probability and updates the current energy.
        // The spin state is copied later by all threads if the move is accepted.
        if (tid == 0) {
            e_new = e_tmp;
            float dE = e_new - chains.energy(chain);
            accept = metropolis_accept(dE, rng, chain, step, 1);

            if (accept) {
                chains.energy(chain) = e_new;
            }
        }

        // Sync here so all threads see the accept/reject decision before copying.
        __syncthreads();

        // If accepted, all threads contribute to copying the trial state into the
        // persistent chain state.
        if (accept) {
            energy_helper.accept_trial_state(s);
        }

        // Sync here so the copied state is complete before the next MCMC step.
        __syncthreads();
    }
}


// Reset states/energies, then measure only the local proposal kernel runtime.
float time_local(ModelBank& models, ChainState& chains, RandomPool& rng) {
    chains.reset_energy();
    reset_states<<<N_CHAINS, BLOCK_THREADS>>>(chains.view());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    local_kernel<<<N_CHAINS, BLOCK_THREADS>>>(models.view(), chains.view(), rng.view());
    CUDA_CHECK(cudaGetLastError());

    return timer.stop_seconds();
}


// Reset states/energies, then measure only the uniform proposal kernel runtime.
float time_uniform(ModelBank& models, ChainState& chains, RandomPool& rng) {
    chains.reset_energy();
    reset_states<<<N_CHAINS, BLOCK_THREADS>>>(chains.view());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    uniform_kernel<<<N_CHAINS, BLOCK_THREADS>>>(models.view(), chains.view(), rng.view());
    CUDA_CHECK(cudaGetLastError());

    return timer.stop_seconds();
}


int main() {
    ModelBank models(SEED);
    ChainState chains(models);
    RandomPool rng(RNG_POOL_SIZE, SEED + 1);

    float local_seconds = time_local(models, chains, rng);
    float uniform_seconds = time_uniform(models, chains, rng);

    std::cout << "SK_N " << SK_N << "\n";
    std::cout << "N_MODELS " << N_MODELS << "\n";
    std::cout << "CHAINS_PER_MODEL " << CHAINS_PER_MODEL << "\n";
    std::cout << "N_CHAINS " << N_CHAINS << "\n";
    std::cout << "N_STEPS " << N_STEPS << "\n";
    std::cout << "RNG_POOL_SIZE " << RNG_POOL_SIZE << "\n";
    std::cout << "local_seconds  " << local_seconds << "\n";
    std::cout << "uniform_seconds " << uniform_seconds << "\n";
    return 0;
}
