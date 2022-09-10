#include <cstdio>
#include <iostream>
#include <vector>
#include <omp.h>
#include <numeric>

using uint = unsigned;
using ldbl = long double;

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

ldbl calc_sum(int rank, uint N, int commsize)
{
  ldbl part = 0.0;

  uint work_amount = N / commsize;
  uint start_i = work_amount * (rank - 1) + 1;

  if (rank == commsize)
    work_amount += N % commsize;

  for (uint i = start_i, end_i = start_i + work_amount; i < end_i; ++i)
    part += 1.0 / i;

  return part;
}

int main( int argc, char *argv[] )
{
  if (argc < 2)
  {
    std::cout << "USAGE: " << argv[0] << " AMOUNT_OF_TERMS\n";
    return 0;
  }

  auto num_terms = std::atoi(argv[1]);

  if (num_terms <= 0)
  {
    std::cout << "Amount of terms must be positive\n";
    return 1;
  }

  auto num_threads = 4;omp_get_max_threads();
  std::vector<ldbl> parts(num_threads);

#pragma omp parallel num_threads(4)
  {
    auto th_id = omp_get_thread_num();
    parts[th_id] = calc_sum(th_id, num_terms, num_threads - 1);
    std::cout << parts[th_id] << std::endl;
  }

  auto sum = std::accumulate(parts.begin(), parts.end(), ldbl(0));

  std::cout << sum << std::endl;
}
