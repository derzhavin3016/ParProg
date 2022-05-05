#include <iostream>
#include <vector>
#include <array>
#include <pthread.h>
#include <sstream>

int NUM_THREADS, N;

using ldbl = long double;

void *calc_part(void *self_num)
{
  ldbl part = 0.0;

  auto id = *reinterpret_cast<int *>(self_num) + 1;
  std::uint64_t work_amount = N / NUM_THREADS;
  std::uint64_t start_i = work_amount * (id - 1) + 1;
  
  if (id == NUM_THREADS)
    work_amount += N % NUM_THREADS;

  for (uint i = start_i, end_i = start_i + work_amount; i < end_i; ++i)
    part += 1.0 / i;

  return new ldbl(part);
}

struct ThInfo
{
  pthread_t th{};
  int id{};
};

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cout << "USAGE: " << argv[0] << " num_threads N" << std::endl;
    return 0;
  }

  NUM_THREADS = std::atoi(argv[1]);
  if (NUM_THREADS <= 0)
  {
    std::cerr << "Amount of threads must be positive" << std::endl;
    return -1;
  }

  N = std::atoi(argv[2]);
  if (N <= 0)
  {
    std::cerr << "N must be positive" << std::endl;
    return -1;
  }

  std::vector<ThInfo> ths;
  ths.resize(NUM_THREADS);
  
  for (int i = 0; i < NUM_THREADS; ++i)
  {
    ths[i].id = i;
    pthread_create(&ths[i].th, nullptr, &calc_part, static_cast<void *>(&ths[i].id));
  }

  ldbl sum = 0;
  for (auto th : ths)
  {
    ldbl *part = nullptr;
    pthread_join(th.th, reinterpret_cast<void **>(&part));
    sum += *part;
    delete part;
  }

  std::cout << sum << std::endl;

  return 0;
}