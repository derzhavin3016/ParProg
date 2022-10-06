#include <cassert>
#include <iostream>
#include <omp.h>

#include "matrix.hh"
#include "timer.hh"

using Mat = linal::Matrix<std::intmax_t>;
using MulFunc = Mat (*)(const Mat &, const Mat &);

Mat mulNaive(const Mat &lhs, const Mat &rhs)
{
  auto res = Mat{lhs.getRows(), rhs.getCols()};
  size_t nrows = res.getRows(), ncols = res.getCols(), com_sz = lhs.getCols();

  for (size_t i = 0; i < nrows; ++i)
    for (size_t j = 0; j < ncols; ++j)
      for (size_t k = 0; k < com_sz; ++k)
        res[i][j] += lhs[i][k] * rhs[k][j];

  return res;
}

Mat mulProm16xTransp(const Mat &lhs, const Mat &rhs)
{
  Mat rhs_t{rhs.Transposing()};
  Mat res{lhs.getRows(), rhs.getCols()};

  size_t res_c = res.getCols(), res_r = res.getRows(), com_sz = lhs.getCols(),
         end_k = com_sz - com_sz % 16;

  for (size_t i = 0; i < res_r; ++i)
    for (size_t j = 0; j < res_c; ++j)
    {
      auto lptr = lhs[i];
      auto rptr = rhs_t[j];
      size_t k = 0;
      for (; k < end_k; k += 16)
        res[i][j] += lptr[k] * rptr[k] + lptr[k + 1] * rptr[k + 1] +
                     lptr[k + 2] * rptr[k + 2] + lptr[k + 3] * rptr[k + 3] +
                     lptr[k + 4] * rptr[k + 4] + lptr[k + 5] * rptr[k + 5] +
                     lptr[k + 6] * rptr[k + 6] + lptr[k + 7] * rptr[k + 7] +
                     lptr[k + 8] * rptr[k + 8] + lptr[k + 9] * rptr[k + 9] +
                     lptr[k + 10] * rptr[k + 10] + lptr[k + 11] * rptr[k + 11] +
                     lptr[k + 12] * rptr[k + 12] + lptr[k + 13] * rptr[k + 13] +
                     lptr[k + 14] * rptr[k + 14] + lptr[k + 15] * rptr[k + 15];

      for (; k < com_sz; ++k)
        res[i][j] += lptr[k] * rptr[k];
    }

  return res;
}

Mat mulOMPNaive(const Mat &lhs, const Mat &rhs)
{
  size_t threads_num = omp_get_max_threads();
  std::cout << "Threads " << threads_num << std::endl;

  auto res = Mat{lhs.getRows(), rhs.getCols()};
  size_t nrows = res.getRows(), ncols = res.getCols(), com_sz = lhs.getCols();

  size_t th_block = nrows / threads_num + 1;

#pragma omp parallel num_threads(threads_num)
  {
    size_t ti = omp_get_thread_num();
    for (size_t i = ti * th_block; i < std::min((ti + 1) * th_block, nrows);
         ++i)
      for (size_t j = 0; j < ncols; ++j)
      {
        for (size_t k = 0; k < com_sz; ++k)
          res[i][j] += lhs[i][k] * rhs[k][j];
      }
  }

  return res;
}

Mat mulOMP16xTransp(const Mat &lhs, const Mat &rhs)
{
  size_t threads_num = omp_get_max_threads();
  std::cout << "Threads " << threads_num << std::endl;

  auto rhs_t = Mat{rhs.Transposing()};
  auto res = Mat{lhs.getRows(), rhs.getCols()};
  size_t nrows = res.getRows(), ncols = res.getCols(), com_sz = lhs.getCols(),
         end_k = com_sz - com_sz % 16, th_block = nrows / threads_num + 1;

  #pragma omp parallel num_threads(threads_num)
  {
    size_t ti = omp_get_thread_num();

    for (size_t i = ti * th_block; i < std::min((ti + 1) * th_block, nrows);
         ++i)
      for (size_t j = 0; j < ncols; ++j)
      {
        auto lptr = lhs[i];
        auto rptr = rhs_t[j];
        size_t k = 0;
        for (; k < end_k; k += 16)
          res[i][j] +=
            lptr[k] * rptr[k] + lptr[k + 1] * rptr[k + 1] +
            lptr[k + 2] * rptr[k + 2] + lptr[k + 3] * rptr[k + 3] +
            lptr[k + 4] * rptr[k + 4] + lptr[k + 5] * rptr[k + 5] +
            lptr[k + 6] * rptr[k + 6] + lptr[k + 7] * rptr[k + 7] +
            lptr[k + 8] * rptr[k + 8] + lptr[k + 9] * rptr[k + 9] +
            lptr[k + 10] * rptr[k + 10] + lptr[k + 11] * rptr[k + 11] +
            lptr[k + 12] * rptr[k + 12] + lptr[k + 13] * rptr[k + 13] +
            lptr[k + 14] * rptr[k + 14] + lptr[k + 15] * rptr[k + 15];

        for (; k < com_sz; ++k)
          res[i][j] += lptr[k] * rptr[k];
      }
  }


  return res;
}

Mat Measure(const Mat &lhs, const Mat &rhs, MulFunc func)
{
  timer::Timer timer;

  auto answ = func(lhs, rhs);

  auto res = static_cast<linal::ldbl>(timer.elapsed_mcs()) / 1'000;
  std::cout << res << " ms" << std::endl;

  return answ;
}

int main()
{
  Mat mat1, mat2, answ;

  std::cin >> mat1 >> mat2 >> answ;

  if (mat1.getCols() != mat2.getRows())
  {
    std::cout << "Incompatible matrix sizes" << std::endl;
    return -1;
  }

  std::cout << "Naive impl\n";
  auto res = Measure(mat1, mat2, mulNaive);
  assert(res == answ);

  std::cout << "Prom 16 + transpose + temp vars\n";
  res = Measure(mat1, mat2, mulProm16xTransp);
  assert(res == answ);

  std::cout << "Naive omp\n";
  res = Measure(mat1, mat2, mulOMPNaive);
  assert(res == answ);

  std::cout << "OMP promp 16 + transpose + temp vars\n";
  res = Measure(mat1, mat2, mulOMP16xTransp);
  assert(res == answ);
}
