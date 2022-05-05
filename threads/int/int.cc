#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

using ldbl = long double;

ldbl func(ldbl x)
{
  return std::sin(1 / x);
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cout << "USAGE " << argv[0] << " num_threads accuracy" << std::endl;
    return 0;
  }

  auto num_threads = std::atoi(argv[1]);
  ldbl accuracy = std::atof(argv[2]);

  std::vector<std::thread> ths;
  ths.reserve(num_threads);

  return 0;
}