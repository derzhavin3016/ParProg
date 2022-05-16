#include <iostream>
#include <vector>
#include <future>
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

inline ldbl run_simp(ldbl a, ldbl b, ldbl step)
{
  std::uint64_t steps = (b - a) / step;
  ldbl res = 0, start = a;

  for (std::uint64_t i = 0; i < steps; ++i)
  {
    res += do_simp(start, start + step);
    start += step;
  }

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
  if (num_threads > static_cast<decltype(num_threads)>(std::thread::hardware_concurrency()))
  {
    std::cerr << "Too many threads" << std::endl;
    return -1;
  }

  ldbl accuracy = std::atof(argv[2]);
  if (accuracy >= 1)
  {
    std::cerr << "Accuracy is too big" << std::endl;
    return -1;
  }

  using ThVal = std::pair<std::thread, std::future<ldbl>>;

  std::vector<ThVal> ths;
  ths.reserve(num_threads);

  auto p_per_thread = PER_AMOUNT / num_threads;
  ldbl inv_start = 1 / A, inv_end = inv_start - PI2 * (p_per_thread + PER_AMOUNT % num_threads);
  inv_end = std::max(inv_end, B);
  std::cout << inv_start << " " << inv_end << std::endl;
  ldbl res = 0;

  for (std::size_t i = 0; i < ths.capacity(); ++i)
  {
    std::promise<ldbl> prom;
    auto fut = prom.get_future();
    auto &&th = std::thread(run_thread, std::move(prom), 1 / inv_start, 1 / inv_end, accuracy);

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

  std::cout << "I = " << res << std::endl;

  return 0;
}