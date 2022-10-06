#!/usr/bin/env python3
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Genrator of matrix for test')
    parser.add_argument('rows1', metavar='ROWS', type=int, help='rows of fst matrix')
    parser.add_argument('cols1', metavar='COLS_N_ROWS', type=int, help='common rows and cols amount')
    parser.add_argument('cols2', metavar='COLS', type=int, help='cols of a sec matrix')
    parser.add_argument('fr', metavar='FROM', type=int, help='start value for generate numbers')
    parser.add_argument('to', metavar='TO', type=int, help='end value for generate numbers')

    args = parser.parse_args()

    if args.rows1<= 0 or args.cols1<= 0 or args.cols2<= 0:
        print('Incorrect size: ' + str(args.size) + '\nTry again')
        return

    matr1 = np.random.randint(args.fr, args.to, (args.rows1, args.cols1))
    matr2 = np.random.randint(args.fr, args.to, (args.cols1, args.cols2))

    print(args.rows1, args.cols1)
    for i in range(args.rows1):
        for j in range(args.cols1):
            print(matr1[i][j], end=' ')
        print()

    print(args.cols1, args.cols2)
    for i in range(args.cols1):
        for j in range(args.cols2):
            print(matr2[i][j], end=' ')
        print()

    mul = np.matmul(matr1, matr2)

    print(args.rows1, args.cols2)
    for i in range(args.rows1):
        for j in range(args.cols2):
            print(mul[i][j], end=' ')
        print()


if __name__ == '__main__':
    main()
