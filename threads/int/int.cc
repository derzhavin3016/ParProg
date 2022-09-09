#include <iostream>
#include <vector>
#include <future>
#include <chrono>
#include <algorithm>
#include <thread>
#include <limits>
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
constexpr ldbl PER_RAT_AMOUNT = (1 / A - 1 / B) / PI2;
constexpr std::uint64_t PER_AMOUNT = static_cast<std::uint64_t>(PER_RAT_AMOUNT);
constexpr ldbl PER_DIFF = PER_RAT_AMOUNT - PER_AMOUNT;

inline ldbl run_simp(ldbl a, ldbl b, ldbl step)
{
  ldbl res = 0;

  for (ldbl start = a; start <= b - step; start += step)
    res += do_simp(start, start + step);

  return res;
}

void run_thread(std::promise<ldbl> res, ldbl a, ldbl b, ldbl accuracy)
{
  auto step = std::cbrt(accuracy);

  ldbl I = 0, Iprev = I, delta = 0;

  do
  {
    I = run_simp(a, b, step);
    delta = I - Iprev;
    Iprev = I;
    step /= 2;
  } while (std::abs(delta) > accuracy);

  res.set_value(I);
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cout << "USAGE " << argv[0] << " num_threads accuracy" << std::endl;
    return 0;
  }

  auto num_threads = std::atoi(argv[1]);
  if (num_threads <= 0)
  {
    std::cerr << "Incorrect threads amount" << std::endl;
    return -1;
  }
  /*
  if (num_threads > static_cast<decltype(num_threads)>(std::thread::hardware_concurrency()))
  {
    std::cerr << "Too many threads" << std::endl;
    return -1;
  }*/

  ldbl accuracy = std::abs(std::atof(argv[2]));
  if (accuracy >= 1)
  {
    std::cerr << "Accuracy is too big" << std::endl;
    return -1;
  }

  using ThVal = std::pair<std::thread, std::future<ldbl>>;

  std::vector<ThVal> ths;
  ths.reserve(num_threads);

  auto p_per_thread = PER_AMOUNT / num_threads;
  ldbl inv_start = 1 / A, inv_end = inv_start - PI2 * (p_per_thread + PER_AMOUNT % num_threads + PER_DIFF);
  inv_end = std::max(inv_end, 1 / B);
  ldbl res = 0;

  auto tic = std::chrono::high_resolution_clock::now();
  for (std::size_t i = 0; i < ths.capacity(); ++i)
  {
    std::promise<ldbl> prom;
    auto fut = prom.get_future();
    auto &&th = std::thread(run_thread, std::move(prom), 1 / inv_start, 1 / inv_end, accuracy / (num_threads * num_threads));

    ths.emplace_back(std::move(th), std::move(fut));

    inv_start = inv_end;
    inv_end -= PI2 * p_per_thread;
    inv_end = std::max(inv_end, 1 / B);
  }

  for (auto &thv : ths)
  {
    res += thv.second.get();
    thv.first.join();
  }
  auto toc = std::chrono::high_resolution_clock::now();

  std::cout.precision(6);//std::numeric_limits<ldbl>::max_digits10);
  std::cout << "I = " << res << std::endl;

  auto mil = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
  std::cout << "Elapsed time " << mil << "ms" << std::endl;

  return 0;
}
