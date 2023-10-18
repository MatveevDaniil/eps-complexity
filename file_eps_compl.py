import math
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev


####################
# binary format tools
####################

def bin_8_bits(number):
    return format(number, '08b')

def readBits(fpath: str) -> Tuple[int]:
    with open('./romeoandJulia.txt', 'rb') as f:
        bytes_repr = map(bin_8_bits, f.read())
    file_bits = [int(fbit) for fbyte in bytes_repr for fbit in fbyte]
    return file_bits

def swapEachKBit(inp_bits: Tuple[int], k: int = 7) -> Tuple[int]:
    out_bits = list(inp_bits)
    for i in range(0, len(inp_bits), k):
        out_bits[i] = 1 - out_bits[i]
    return tuple(out_bits)


####################
# calc eps-complexity 
####################

def L1(x: np.array, y: np.array) -> np.array:
    return np.sum(abs(x - y))

def get_S_k(t: np.array, k: int) -> np.array:
    return t.reshape((-1, k)).T

def calc_eps_on_S_k(t: np.array, x: np.array, k: int, norm: callable = L1) -> float:
    S_k = get_S_k(t, k)
    x_S_k = get_S_k(x, k)
    errors = []
    for i in range(k):
        error_i = float('inf')
        for deg in (1, 2, 3, 4):
            spl = splrep(S_k[i], x_S_k[i], k=deg)
            dropped_approx = splev(np.delete(S_k, i, axis=0).T.flatten(), spl)
            dropped_value = np.delete(x_S_k, i, axis=0).T.flatten()
            error_i = min(error_i, norm(dropped_approx, dropped_value))
        errors.append(error_i)
    return np.mean(errors)

def calc_eps(x: np.array, norm: callable = L1, k_list = [2, 3, 4, 5]) -> Dict[int, float]:
    eps = {}
    new_len = len(x) - len(x) % math.lcm(*k_list, 16)
    x = x[:new_len]
    t = np.arange(0, new_len // 16, 0.0625)
    for k in k_list:
        eps[k] = round(calc_eps_on_S_k(t, x, k, norm), 2)
    return eps


####################
# test work
####################

def main():
    file_bits = readBits('./romeoandJulia.txt')
    zero_eps = calc_eps(np.array(file_bits))
    double_swap = swapEachKBit(file_bits, k=1)
    double_swap = swapEachKBit(double_swap, k=1)
    __zero_eps = calc_eps(np.array(double_swap))
    print(file_bits[:10])
    print(zero_eps)
    print(__zero_eps)
    print('=' * 20)
    plt.scatter(zero_eps.keys(), zero_eps.values(), label='0')
    for k in [1, 2, 4, 7, 25, 100]:
        new_bits = swapEachKBit(file_bits, k=k)
        print(k)
        print(file_bits[:10])
        changed_bits = tuple(int(file_bits[i] == new_bits[i]) for i in range(len(file_bits)))
        eps = calc_eps(np.array(changed_bits))
        print(eps)
        print('=' * 20)
        plt.scatter(eps.keys(), eps.values(), label=f'{k}')
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()