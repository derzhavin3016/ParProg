#include <iostream>
#include <vector>
#include <array>
#include <pthread.h>
#include <sstream>

int NUM_THREADS, N;

using ldbl = long double;

int message = 0;
pthread_mutex_t pmut = PTHREAD_MUTEX_INITIALIZER;

void *do_ring(void *self_num)
{
  std::ostringstream ost;
  ost << "Thread " << *reinterpret_cast<int *>(self_num) << " msg ";

  pthread_mutex_lock(&pmut);
  auto cur_msg = message++;
  pthread_mutex_unlock(&pmut);

  ost << cur_msg << std::endl;
  std::cout << ost.str();

  return nullptr;
}

struct ThInfo
{
  pthread_t th{};
  int id{};
};

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "USAGE: " << argv[0] << " num_threads" << std::endl;
    return 0;
  }

  NUM_THREADS = std::atoi(argv[1]);
  if (NUM_THREADS <= 0)
  {
    std::cerr << "Amount of threads must be positive" << std::endl;
    return -1;
  }

  std::vector<ThInfo> ths;
  ths.resize(NUM_THREADS);
  
  pthread_mutex_unlock(&pmut);

  for (int i = 0; i < NUM_THREADS; ++i)
  {
    ths[i].id = i;
    pthread_create(&ths[i].th, nullptr, &do_ring, static_cast<void *>(&ths[i].id));
  }

  for (auto th : ths)
    pthread_join(th.th, nullptr);

  return 0;
}