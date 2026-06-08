#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int n;
vector<double> h;
vector<double> J;
vector<double> betas = {0.0, 1.0 / 32.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 8.0, 3.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 2.0, 5.0 / 8.0, 3.0 / 4.0, 13.0 / 16.0, 7.0 / 8.0, 15.0 / 16.0, 31.0 / 32.0, 1.0, 33.0 / 32.0, 17.0 / 16.0, 9.0 / 8.0, 19.0 / 16.0, 5.0 / 4.0, 11.0 / 8.0, 3.0 / 2.0, 7.0 / 4.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0};

int jidx(int i, int j) {
    return i * n - i * (i + 1) / 2 + (j - i - 1);
}

double Jij(int i, int j) {
    return J[jidx(i, j)];
}

double energy(uint64_t x) {
    int8_t s[64];
    double E = 0.0;

    for(int i = 0; i < n; i++) {
        s[i] = ((x >> i) & 1ULL) ? -1 : 1;
    }

    for(int i = 0; i < n; i++) {
        E -= h[i] * s[i];
    }

    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            E -= Jij(i, j) * s[i] * s[j];
        }
    }

    return E;
}

void fill_vectors(char* filename) {
    ifstream in(filename);
    if(!in) {
        cerr << "cannot open input file\n";
        exit(1);
    }

    h.assign(n, 0.0);
    J.assign(n * (n - 1) / 2, 0.0);

    for(int i = 0; i < n; i++) {
        in >> h[i];
    }

    for(int i = 0; i < n * (n - 1) / 2; i++) {
        in >> J[i];
    }
}

double find_Emin(uint64_t N) {
    double Emin = numeric_limits<double>::infinity();

    #pragma omp parallel
    {
        double local_min = numeric_limits<double>::infinity();

        #pragma omp for schedule(static)
        for(int64_t x = 0; x < (int64_t)N; x++) {
            local_min = min(local_min, energy((uint64_t)x));
        }

        #pragma omp critical
        {
            Emin = min(Emin, local_min);
        }
    }

    return Emin;
}

vector<double> exact_variances_streaming(uint64_t N, double Emin) {
    int B = (int)betas.size();
    vector<long double> Z(B, 0.0L);
    vector<long double> M1(B, 0.0L);
    vector<long double> M2(B, 0.0L);

    #pragma omp parallel
    {
        vector<long double> local_Z(B, 0.0L);
        vector<long double> local_M1(B, 0.0L);
        vector<long double> local_M2(B, 0.0L);

        #pragma omp for schedule(static)
        for(int64_t x = 0; x < (int64_t)N; x++) {
            double e = energy((uint64_t)x);
            long double de = (long double)e - (long double)Emin;

            for(int b = 0; b < B; b++) {
                long double w = expl(-(long double)betas[b] * de);
                local_Z[b] += w;
                local_M1[b] += w * (long double)e;
                local_M2[b] += w * (long double)e * (long double)e;
            }
        }

        #pragma omp critical
        {
            for(int b = 0; b < B; b++) {
                Z[b] += local_Z[b];
                M1[b] += local_M1[b];
                M2[b] += local_M2[b];
            }
        }
    }

    vector<double> vars(B);

    for(int b = 0; b < B; b++) {
        long double mean = M1[b] / Z[b];
        long double mean2 = M2[b] / Z[b];
        vars[b] = (double)(mean2 - mean * mean);
    }

    return vars;
}

int main(int argc, char** argv) {
    cout << unitbuf;

    if(argc < 3) {
        cerr << "usage: ./executable <n> <filename.txt>\n";
        return 1;
    }

    n = stoi(argv[1]);

    if(n <= 0 || n > 25) {
        cerr << "this exact streaming code allows 1 <= n <= 25\n";
        return 1;
    }

    fill_vectors(argv[2]);

    uint64_t N = 1ULL << n;
    double Emin = find_Emin(N);
    vector<double> vars = exact_variances_streaming(N, Emin);

    cout << setprecision(17);

    for(size_t i = 0; i < betas.size(); i++) {
        cout << betas[i] << " " << vars[i] << "\n";
    }

    return 0;
}