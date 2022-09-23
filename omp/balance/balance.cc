#include <iostream>
#include <omp.h>
#include <array>
#include <sstream>
#include <algorithm>

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

#define FOR_CYCLE(END)                                                         \
  for (int i = 0; i < (END); ++i)                                              \
  {                                                                            \
    std::ostringstream ss;                                                     \
    int tid = omp_get_thread_num();                                            \
    task_dist[tid][i] = true;                                                  \
  }

constexpr int end_cycle = 65;
using MapTh = std::array<std::array<bool, end_cycle>, 4>;

void clearMap(MapTh &map)
{
  for (auto &sub : map)
    std::fill_n(sub.begin(), sub.size(), false);
}

void printMap(const MapTh &map)
{
  for (size_t tid = 0; tid < map.size(); ++tid)
  {
    std::cout << "Thread " << tid << " : ";
    for (size_t iter = 0; iter < map[tid].size(); ++iter)
      if (map[tid][iter])
        std::cout << iter << ' ';
    std::cout << std::endl;
  }
}

int main()
{
  MapTh task_dist{};

  omp_set_num_threads(4);
  std::cout << "dynamic chunk 1" << std::endl;

  #pragma omp parallel for schedule(dynamic, 1)
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "dynamic chunk 4" << std::endl;

  #pragma omp parallel for schedule(dynamic, 4)
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "static chunk 1" << std::endl;

  #pragma omp parallel for schedule(static, 1)
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "static chunk 4" << std::endl;

  #pragma omp parallel for schedule(static, 4)
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "guided chunk 1" << std::endl;

  #pragma omp parallel for schedule(guided, 1)
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "guided chunk 4" << std::endl;

  #pragma omp parallel for schedule(guided, 4)
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "default chunk 1" << std::endl;

  #pragma omp parallel for
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "default chunk 4" << std::endl;

  #pragma omp parallel for
    FOR_CYCLE(end_cycle)

  printMap(task_dist);
  clearMap(task_dist);
  std::cout << "END\n";
}
