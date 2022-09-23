#include <iostream>
#include <sstream>
#include <omp.h>

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

int main()
{
  omp_set_nested(1);
  #pragma omp parallel num_threads(3)
  {
    auto threads1 = omp_get_num_threads();
    auto id1 = omp_get_thread_num();

    #pragma omp parallel num_threads(2)
    {
      auto threads2 = omp_get_num_threads();
      auto id2 = omp_get_thread_num();

      #pragma omp parallel num_threads(2)
      {
        auto threads3 = omp_get_num_threads();
        auto id3 = omp_get_thread_num();


        std::ostringstream ss;
        ss << "=========================\n";
        ss << "From 1 num_threads = " << threads1 << " thread_num = " << id1 << std::endl;
        ss << "From 2 num_threads = " << threads2 << " thread_num = " << id2 << std::endl;
        ss << "From 3 num_threads = " << threads3 << " thread_num = " << id3 << std::endl;
        ss << "=========================\n";

        std::cout << ss.str();
      }
    }
  }
}
