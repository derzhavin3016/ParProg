#include <iostream>
#include <vector>
#include <array>
#include <pthread.h>
#include <sstream>

int NUM_THREADS;

void *print_hello(void *self_num)
{
  std::ostringstream ost;
  
  ost << "I'm " << *reinterpret_cast<int *>(self_num) << " out of " << NUM_THREADS << std::endl;
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
  
  for (int i = 0; i < NUM_THREADS; ++i)
  {
    ths[i].id = i;
    pthread_create(&ths[i].th, nullptr, &print_hello, static_cast<void *>(&ths[i].id));
  }

  for (auto th : ths)
    pthread_join(th.th, nullptr);

  return 0;
}