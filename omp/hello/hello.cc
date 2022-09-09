#include <iostream>
#include <sstream>
#include <omp.h>

int main()
{
  auto max_threads = omp_get_max_threads();

#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
    std::stringstream ss;
    ss << "Hello, I'm thread " << thread_id
       << " out of " << max_threads << "\n";

    std::cout << ss.str();
  }
}
