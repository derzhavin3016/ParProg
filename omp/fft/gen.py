#!/usr/bin/env python3


import argparse
import numpy as np

FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
}

def print_compl_arr(arr):
    tup = [f"({np.real(elem)}, {np.imag(elem)})" for elem in arr]
    print(" ".join(tup))

def main():
    parser = argparse.ArgumentParser(description="Tool to generate FFT tests")

    parser.add_argument("func", help="Base function", type=str, choices=FUNCS.keys())
    parser.add_argument("N", help="Size of vector", type=int)

    args = parser.parse_args()

    assert (
        args.N > 1 and args.N & (args.N - 1) == 0
    ), "N should be a non-zero power of 2"

    k = np.arange(args.N)
    arg = 2 * np.pi * k / len(k)
    if args.func == "exp":
        arg = arg * 1j
    x = FUNCS[args.func](arg)

    X = np.fft.fft(x)

    print(len(x))
    print_compl_arr(x)
    print(len(X))
    print_compl_arr(X)


if "__main__" == __name__:
    main()
