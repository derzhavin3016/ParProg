#include <iostream>
#include <sstream>
#include <random>
#include <omp.h>

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

uint importantCounter = 0;
#pragma omp threadprivate(importantCounter)

void print_thread(uint cnt)
{
  std::ostringstream ss;
  ss << "Thread " << omp_get_thread_num() << " importantCounter = " << cnt << std::endl;

  std::cout << ss.str();
}

int main()
{
  importantCounter = 3;

#if defined(WITH_COPYIN)
  std::cout << "Copyin case:\n";
  #pragma omp parallel copyin(importantCounter)
#else
  std::cout << "No copyin case:\n";
  #pragma omp parallel
#endif
    print_thread(importantCounter);

  std::cout << "At end " << importantCounter << std::endl << std::endl;
}
