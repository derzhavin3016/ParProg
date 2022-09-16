#include <iostream>
#include <sstream>
#include <omp.h>

using uint = unsigned;
using ldbl = long double;

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

int main()
{
  uint msg = 0;
  auto max_th = omp_get_max_threads();

#pragma omp parallel for ordered schedule(dynamic)
  for (auto th_id = 0; th_id != max_th; ++th_id)
  {
    std::ostringstream ss;
    ss << "Thread " << th_id << " msg ";
    #pragma omp ordered
    {
      ss << msg++;
    }
    ss << std::endl;
    std::cout << ss.str();
  }

}
