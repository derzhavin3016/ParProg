#include <iostream>
#include <sstream>
#include <random>
#include <omp.h>

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

uint importantCounter = 0;
#pragma omp threadprivate(importantCounter)

int main()
{
  importantCounter = 3;
  #pragma omp parallel //copyin(importantCounter)
  {
    #pragma omp master
      importantCounter += 228;

    importantCounter += omp_get_thread_num();

    std::ostringstream ss;
    ss << "Thread " << omp_get_thread_num() << " importantCounter = " << importantCounter << std::endl;

    std::cout << ss.str();
  }

  std::cout << "At end " << importantCounter << std::endl;

}
