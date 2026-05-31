from monaqa2.data.filename import TIMING_CPU_FOLDER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear


def cpu_time_per_local_move(n: int) -> float:
    """
    Return the fitted CPU time for one local move on a system of n spins.

    :param n: Number of spins.
    :return: Estimated seconds per local move on a system of n spins.
    """
    # if I have two points (x=10, y=3.33 * 21) and (x=30, y=3.33 * 32) what is the rule y(x) = a * log(x) that best fit this line?
    # y(x) = a log(x)
    # a = ((3.33 * 21) log(10) + (3.33 * 32) log(30)) / (log(10)^2 + log(30)^2)
    # a approx 30
    return 30e-9 * np.log(n)


def cpu_time_per_uniform_move(n: int) -> float:
    """
    Return the fitted CPU time for one uniform move on a system of n spins.

    :param n: Number of spins.
    :return: Estimated seconds per uniform move on a system of n spins.
    """
    return 0
