#ifndef __OMP_MATMUL_MATMUL_HH__
#define __OMP_MATMUL_MATMUL_HH__

#include <algorithm>
#include <array>
#include <cassert>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "matrix.hh"
#include "timer.hh"

namespace mul
{

using Mat = linal::Matrix<std::int32_t>;
using MulFunc = Mat (*)(const Mat &, const Mat &);

Mat mulNaive(const Mat &lhs, const Mat &rhs)
{
  auto res = Mat{lhs.getRows(), rhs.getCols()};
  std::size_t nrows = res.getRows(), ncols = res.getCols(),
              com_sz = lhs.getCols();

  for (std::size_t i = 0; i < nrows; ++i)
    for (std::size_t j = 0; j < ncols; ++j)
      for (std::size_t k = 0; k < com_sz; ++k)
        res[i][j] += lhs[i][k] * rhs[k][j];

  return res;
}

Mat mulProm16xTransp(const Mat &lhs, const Mat &rhs)
{
  Mat rhs_t{rhs.Transposing()};
  Mat res{lhs.getRows(), rhs.getCols()};

  std::size_t res_c = res.getCols(), res_r = res.getRows(),
              com_sz = lhs.getCols(), end_k = com_sz - com_sz % 16;

  for (std::size_t i = 0; i < res_r; ++i)
    for (std::size_t j = 0; j < res_c; ++j)
    {
      auto lptr = lhs[i];
      auto rptr = rhs_t[j];
      std::size_t k = 0;
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
  std::size_t threads_num = omp_get_max_threads();
  std::cout << "Threads " << threads_num << std::endl;

  auto res = Mat{lhs.getRows(), rhs.getCols()};
  std::size_t nrows = res.getRows(), ncols = res.getCols(),
              com_sz = lhs.getCols();

  std::size_t th_block = nrows / threads_num + 1;

#pragma omp parallel num_threads(threads_num)
  {
    std::size_t ti = omp_get_thread_num();
    for (std::size_t i = ti * th_block;
         i < std::min((ti + 1) * th_block, nrows); ++i)
      for (std::size_t j = 0; j < ncols; ++j)
      {
        for (std::size_t k = 0; k < com_sz; ++k)
          res[i][j] += lhs[i][k] * rhs[k][j];
      }
  }

  return res;
}

Mat mulOMP16xTransp(const Mat &lhs, const Mat &rhs)
{
  std::size_t threads_num = omp_get_max_threads();
#if defined(CompareWays)
  std::cout << "Threads " << threads_num << std::endl;
#endif

  auto rhs_t = Mat{rhs.Transposing()};
  auto res = Mat{lhs.getRows(), rhs.getCols()};
  std::size_t nrows = res.getRows(), ncols = res.getCols(),
              com_sz = lhs.getCols(), end_k = com_sz - com_sz % 16,
              th_block = nrows / threads_num + 1;

#pragma omp parallel num_threads(threads_num)
  {
    std::size_t ti = omp_get_thread_num();

    for (std::size_t i = ti * th_block;
         i < std::min((ti + 1) * th_block, nrows); ++i)
      for (std::size_t j = 0; j < ncols; ++j)
      {
        auto lptr = lhs[i];
        auto rptr = rhs_t[j];
        std::size_t k = 0;
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

Mat mulProm8xTranspIntr(const Mat &lhs, const Mat &rhs)
{
  Mat rhs_t{rhs.Transposing()};
  Mat res{lhs.getRows(), rhs.getCols()};

  std::size_t res_c = res.getCols(), res_r = res.getRows(),
              com_sz = lhs.getCols(), end_k = com_sz - com_sz % 8;

  for (std::size_t i = 0; i < res_r; ++i)
    for (std::size_t j = 0; j < res_c; ++j)
    {
      const auto lptr = lhs[i];
      const auto rptr = rhs_t[j];

      auto lp_intr = reinterpret_cast<const __m256i *>(lptr);
      auto rp_intr = reinterpret_cast<const __m256i *>(rptr);

      std::size_t k = 0;
      auto sum = _mm256_setzero_si256();
      for (; k < end_k; k += 8)
      {
        auto lhs_v = _mm256_loadu_si256(lp_intr++);
        auto rhs_v = _mm256_loadu_si256(rp_intr++);

        auto mul_v = _mm256_mullo_epi32(lhs_v, rhs_v);

        sum = _mm256_add_epi32(sum, mul_v);
      }

      auto swp128 = _mm256_permute2x128_si256(sum, sum, 1);
      auto sum128 = _mm256_castsi256_si128(_mm256_add_epi32(sum, swp128));

      auto swp64 = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2));
      auto sum64 = _mm_add_epi32(swp64, sum128);

      auto swp32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE2(0, 1));
      auto sum32 = _mm_add_epi32(swp32, sum64);

      std::int32_t res_sum = _mm_cvtsi128_si32(sum32);

      for (; k < com_sz; ++k)
        res_sum += lptr[k] * rptr[k];

      res[i][j] = res_sum;
    }

  return res;
}

std::pair<Mat, linal::ldbl> Measure(const Mat &lhs, const Mat &rhs,
                                    MulFunc func)
{
  timer::Timer timer;

  auto answ = func(lhs, rhs);

  auto res = static_cast<linal::ldbl>(timer.elapsed_mcs()) / 1'000;

  return {answ, res};
}
} // namespace mul

#endif /* __OMP_MATMUL_MATMUL_HH__ */
