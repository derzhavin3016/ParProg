#include <iostream>
#include <omp.h>
#include <sstream>

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

#define FOR_CYCLE(END)                                                         \
  for (int i = 0; i < (END); ++i)                                              \
  {                                                                            \
    std::ostringstream ss;                                                     \
    int tid = omp_get_thread_num();                                            \
    ss << "Thread " << tid << " iteration " << i << std::endl;                 \
    std::cout << ss.str();                                                     \
  }

int main()
{
  constexpr int end_cycle = 65;
  omp_set_num_threads(4);
  std::cout << "dynamic chunk 1" << std::endl;

  #pragma omp parallel for schedule(dynamic, 1)
    FOR_CYCLE(end_cycle)

  std::cout << "dynamic chunk 4" << std::endl;

  #pragma omp parallel for schedule(dynamic, 4)
    FOR_CYCLE(end_cycle)

  std::cout << "static chunk 1" << std::endl;

  #pragma omp parallel for schedule(static, 1)
    FOR_CYCLE(end_cycle)

  std::cout << "static chunk 4" << std::endl;

  #pragma omp parallel for schedule(static, 4)
    FOR_CYCLE(end_cycle)

  std::cout << "guided chunk 1" << std::endl;

  #pragma omp parallel for schedule(guided, 1)
    FOR_CYCLE(end_cycle)

  std::cout << "guided chunk 4" << std::endl;

  #pragma omp parallel for schedule(guided, 4)
    FOR_CYCLE(end_cycle)
  std::cout << "default chunk 1" << std::endl;

  #pragma omp parallel for
    FOR_CYCLE(end_cycle)

  std::cout << "default chunk 4" << std::endl;

  #pragma omp parallel for
    FOR_CYCLE(end_cycle)

  std::cout << "END\n";
}
