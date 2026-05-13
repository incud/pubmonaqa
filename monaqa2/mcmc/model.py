import numpy as np
import scipy.linalg as la


# ========================================================================
# ====== UTILITIES FUNCTION TO SUPPORT THE ISING MODEL GENERATION ========
# ========================================================================

def int_to_bin(i, n):
    '''Convert a given integer to a bitstring of fixed length using 0 -> 0..00 convention.
    The function is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    return bin(i)[2:].zfill(n)


def int_to_bin_arr(i, n):
    '''Convert a given integer to a bitstring of fixed length using 0 -> 0..00 convention.
    The function is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    return np.array([int(b) for b in int_to_bin(i, n)])


def bin_to_int(s):
    '''Convert a given bitstring to an integer.
    The function is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    return int(s,2)


def bin_to_spin(b):
    '''Convert a binary string to a spin configuration using 0 -> +1 convention.
    The function is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    if len(b) == 1: 
        return 1-2*int(b)
    else:
        s = [1-2*int(c) for c in b]
        return s


def int_to_spin(i, n):
    '''Convert an integer to a spin configuration of specified length.The function is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    b = int_to_bin(i, n)
    return bin_to_spin(b)


def generate_random_J(n, seed=None):
    '''Generate a random symmetric coupling matrix for n sites, sampled from N(0,1).
    The function is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    rng = np.random.default_rng(seed)
    J = np.triu(rng.standard_normal((n, n)), k=1)  # No self-interactions
    J = J + J.T  # Ensure symmetry
    return J


def generate_random_h(n, seed=None):
    '''Generate random external fields for n sites, sampled from N(0,1).
    The function is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    rng = np.random.default_rng(None if seed is None else seed + int((n + 1) * n / 2))
    return rng.standard_normal(n)


# ========================================================================
# ==================== ISING MODEL AND RANDOM ISING MODEL ================
# ========================================================================


class IsingModel:
    '''
    Represents an Ising model with given parameters J and h.    
    The class is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    def __init__(self, J, h):
        self.h = h
        self.n = h.size
        self.J = J

        assert np.allclose(J.T, J), 'J must be symmetric.'
        assert np.all(np.diag(J) == 0), 'Diagonal elements of J must be zero.'

        # Rescale couplings to keep energies comparable across instance sizes
        self.alpha = np.sqrt(self.n / (0.5*la.norm(J, ord='fro')**2 + la.norm(h, ord=2)**2))
        self.J_rescaled = self.J * self.alpha 
        self.h_rescaled = self.h * self.alpha

        # Calculate and store this only if necessary
        self.__gs = None
        self.__gs_energy = None
        self.__gs_deg = None
        self.__E = None
        self.__E_rescaled = None

    @property
    def energies(self) -> np.ndarray:
        if self.__E is None:
            self.__E = np.zeros((2**self.n))
            for i in range(2**self.n):
                s = int_to_spin(i, self.n)
                self.__E[i] = -0.5*(s @ self.J @ s) - s @ self.h
        return self.__E
    
    @property
    def energies_rescaled(self) -> np.ndarray:
        if self.__E_rescaled is None:
            self.__E_rescaled = self.energies.copy() * self.alpha
        return self.__E_rescaled
    
    @property
    def ground_state(self) -> str:
        if self.__gs is None or self.__gs_energy is None:
            idx = np.argmin(self.energies)
            self.__gs = int_to_bin(idx, self.n)
            self.__gs_energy = self.energies[idx]
        return self.__gs
    
    @property
    def ground_state_energy(self) -> str:
        if self.__gs is None or self.__gs_energy is None:
            idx = np.argmin(self.energies)
            self.__gs = int_to_bin(idx, self.n)
            self.__gs_energy = self.energies[idx]
        return self.__gs

    @property
    def ground_state_degeneracies(self) -> int:
        if self.__gs_deg is None:
            self.__gs_deg = np.count_nonzero(np.isclose(self.energies, self.__gs_energy))
        return self.__gs_deg

    @classmethod
    def from_coefficients(cls, n, coefficients):
        '''
        Instantiates an Ising model from an array of coefficients.
        
        Parameters:
        - coefficients (numpy.ndarray): 1D array of n + n(n-1)/2 coefficients, where the first n elements
                                        correspond to h (external fields) and the remaining correspond to
                                        the upper triangular elements of the symmetric interaction matrix J.
        '''
        #  Check that the length of the coefficients array is consistent
        assert len(coefficients) == n+(n*(n-1))//2, \
            f"Expected {n+(n*(n-1))//2} coefficients, but got {len(coefficients)}."

        h = np.asarray(coefficients[:n])
        J_upper_tri = coefficients[n:]
        J = np.zeros((n, n))

        # Fill the upper triangular part of J with the interaction coefficients
        upper_tri_indices = np.triu_indices(n, k=1)
        J[upper_tri_indices] = J_upper_tri
        J += J.T
        return cls(J, h)


class RandomIsingModel(IsingModel):
    '''
    Represents a randomly generated Ising model with a given number of qubits.
    The class is imported and edited from https://github.com/petr-ivashkov/quantum-mcmc
    '''
    def __init__(self, n, local_fields=True, seed=None):
        h = generate_random_h(n, seed=seed) if local_fields else np.zeros(n)
        J = generate_random_J(n, seed=seed)
        super().__init__(J, h)
