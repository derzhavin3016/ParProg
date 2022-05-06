#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <cmath>

using ldbl = long double;

inline ldbl func(ldbl x)
{
  return std::sin(1 / x);
}

inline ldbl do_simp(ldbl a, ldbl b)
{
  return (b - a) / 6 * (func(a) + 4 * func((a + b) / 2) + func(b));
}

constexpr ldbl A = 0.001;
constexpr ldbl B = 1;
constexpr ldbl PI = 3.14159265358979323846;
constexpr ldbl PI2 = 2 * PI;
constexpr std::uint64_t PER_AMOUNT = (1 / A - 1 / B) / PI2;

void run_thread(ldbl a, ldbl b)
{
  auto opt_step = 0;
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

  auto p_per_thread = PER_AMOUNT / num_threads;
  ldbl inv_start = 1 / A, inv_end = inv_start - PI2 * (p_per_thread + PER_AMOUNT % num_threads);
  inv_end = std::max(inv_end, 1 / B);

  for (std::size_t i = 0; i < ths.capacity(); ++i)
  {
    ths.emplace_back(run_thread, 1 / inv_start, 1 / inv_end);
    inv_start = inv_end;
    inv_end -= PI2 * p_per_thread;
    inv_end = std::max(inv_end, 1 / B);
  }

  for (auto &th : ths)
    th.join();


  return 0;
}