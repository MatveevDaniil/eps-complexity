import os
import json
import numpy as np
from typing import Tuple
from scipy.interpolate import splrep, splev, interp1d
import matplotlib.pyplot as plt


def L1(x: np.array, y: np.array) -> np.array:
    return np.sum(abs(x - y))

def L2(x: np.array, y: np.array) -> np.array:
    return ((x - y) ** 2).sum() ** 0.5

def get_S_k(t: np.array, k: int) -> np.array:
    return t.reshape((-1, k)).T

def get_S_k_i(t: np.array, k: int, i: int) -> tuple[np.array]:
    mask = np.full_like(t, False, dtype=bool)
    mask[i::k] = True
    return t[mask], t[~mask]

def x_sinuses(t: np.array) -> np.array:
    w = 241, 251, 257, 263, 269, 271, 277, 281, 283, 293
    x = np.zeros_like(t)
    for w_i in w:
        x += np.sin(w_i * t)
    return x

def calc_eps_on_S_k(
        t: np.array, 
        k: int, 
        x: callable, 
        norm: callable = L1,
        log_dict: str = None
) -> float:
    errors = []
    for i in range(k):
        S_k_i, S_k_not_i = get_S_k_i(t, k, i)
        x_S_k_i, dropped_value = x(S_k_i), x(S_k_not_i)
        
        error_i = float('inf')

        # piecewise constants
        spl0_zero = interp1d(S_k_i, x_S_k_i, kind='zero', fill_value="extrapolate")
        dropped_approx = spl0_zero(S_k_not_i)
        error_i = min(error_i, norm(dropped_approx, dropped_value))

        spl0_nearest = interp1d(S_k_i, x_S_k_i, kind='nearest', fill_value="extrapolate")
        dropped_approx = spl0_nearest(S_k_not_i)
        error_i = min(error_i, norm(dropped_approx, dropped_value))
        
        # splines power 1-3
        for deg in (1, 2, 3, 4):
            spl = splrep(S_k_i, x_S_k_i, k=deg)

            dropped_approx = splev(S_k_not_i, spl)
            error_i = min(error_i, norm(dropped_approx, dropped_value))

        errors.append(error_i)
    return np.mean(errors)

def calc_A_B(
        t: np.array, 
        x: callable, 
        norm: callable = L1,
        log_dict: str = None,
        drop_ratio: list[int] = [2, 3, 4, 5, 6]
) -> Tuple[float]:
    log_dict['t'] = t.tolist() #
    log_dict['x'] = x(t).tolist() #
    log_dict['(1/drop_ratio, error)'] = [] #
    log_S, log_eps = [], []
    for k in drop_ratio:
        error_S_k = calc_eps_on_S_k(t, k, x, norm, log_dict)
        log_dict['(1/drop_ratio, error)'].append((k, error_S_k)) #
        log_S.append(np.log(1 / k))
        log_eps.append(np.log(error_S_k))
    _log_S = np.vstack([log_S, np.ones_like(log_S)]).T
    (B, A), _, _, _ = np.linalg.lstsq(_log_S, log_eps, rcond=None)
    log_dict['(log(drop_ratio), log_error)'] = [list(zip(log_S, log_eps))]
    log_dict['A'] = A
    log_dict['B'] = B
    return (A, B)

def limit_A(
        x: callable, 
        grid_steps: Tuple[float], 
        norm: callable = L1,
        log_file_dir: str = None
):
    collect_A = []
    collect_B = []
    for h in grid_steps:
        log_dict = {}
        t = np.arange(0, 1, step=h)
        n = len(t)
        A, B = calc_A_B(t, x, norm, log_dict)
        collect_A.append(- A / np.log(n))
        collect_B.append(B)
        log_file_name = f'{log_file_dir}/h={h}.json'
        log_dict['-A / log(n)'] = - A / np.log(n)
        log_dict['h'] = h
        log_dict['n'] = n
        with open(log_file_name, 'w') as f:
            json.dump(log_dict, f)
    return collect_A, collect_B

def main():
    grid_steps = [0.01, 0.001, 0.0008, 0.0005, 0.0004, 0.0002, 0.0001, 0.00005, 0.00004, 0.00002, 0.00001]
    log_file_dir = '/Users/daniil/sfsu/DrPiryatinska/logs_sinuses_241_293'
    if not os.path.exists(log_file_dir):
        os.mkdir(log_file_dir)
    A, B = limit_A(x_sinuses, grid_steps, log_file_dir=log_file_dir)

    x_p = -np.log10(grid_steps)
    plt.scatter(x_p, A,  label=r'-$ \frac{\hat{A}}{\ln{n}}$', )
    plt.xlabel('$-\log_{10}{h}$')
    plt.ylabel(r'$\frac{-A}{\ln{n}}$')
    plt.legend()
    plt.title('holder exponent through eps-complexity \n  $range_x=[0, 1], h_j \in [10^{-2}, 10^{-5}] $\n $f(x)=\sum_k \sin{(w_k x)}$ \n $w_k$ - primes from 241 to 293')
    plt.tight_layout()
    plt.savefig('./helder exponent for sin.png', dpi=300) 
    plt.show()

if __name__ == "__main__":
    main()