#include "erat.hh"

Numbers seqEr_res{};

bool isPrime(std::uint64_t num)
{
  if (num == 1)
    return false;
  if (num == 2)
    return true;
  for (std::uint64_t i = 2; i <= num >> 1; ++i)
    if (num % i == 0)
      return false;

  return true;
}

bool isPrimes(const Numbers &numbers)
{
  return std::all_of(numbers.begin(), numbers.end(), isPrime);
}

template <typename T>
std::pair<long double, Numbers> measure(std::uint64_t num, T func) requires
  std::is_convertible_v<T, EratNaiveFunc> || std::is_convertible_v<T, EratFunc>
{
  timer::Timer tim;

  auto res = func(num);

  auto elapsed = tim.elapsed_mcs();

  Numbers nums;
  if constexpr (std::is_convertible_v<T, EratNaiveFunc>)
    nums = std::move(res);
  else if constexpr (std::is_convertible_v<T, EratFunc>)
    nums = fromSieve(num, res);
  else
    throw std::runtime_error("");

  return {elapsed * 1e-3, nums};
}

template <typename T>
void checkPrint(std::uint64_t num, T func, std::string_view name)
{
  auto [ms, res] = measure<T>(num, func);
  bool isSeq = name == "seqErat";
  if (isSeq)
    seqEr_res = res;
  bool passed = isSeq ? isPrimes(res) : seqEr_res == res;

  if (num < 2000)
  {
    std::for_each(res.begin(), res.end(),
                  [](auto n) { std::cout << n << " "; });
    std::cout << std::endl;
  }

  std::cout << name << std::endl;
  std::cout << ms << " ms" << std::endl;
  std::cout << "Status: "
            << (passed ? "\033[1;32mPassed\033[0m" : "\033[1;31mFailed\033[0m")
            << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " N" << std::endl;
    return 1;
  }

  auto N = std::atoll(argv[1]);
  if (N <= 0)
  {
    std::cout << "You must provide positive number" << std::endl;
    return 1;
  }

#define CHECK_ERAT(num, func) checkPrint((num), (func), #func)
#define CHECK_N_ERAT(func) CHECK_ERAT(N, func)

  CHECK_N_ERAT(seqErat);
  CHECK_N_ERAT(seqOddErat);
  CHECK_N_ERAT(parOddErat);

#undef CHECK_ERAT
#undef CHECK_N_ERAT
}
