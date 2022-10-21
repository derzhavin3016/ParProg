#include "fft.hh"

namespace fft
{
void inputVec(Vec &vec, std::istream &ist /* = std::cin */)
{
  std::size_t N = 0;
  ist >> N;

  vec.resize(N);
  for (auto &num : vec)
    ist >> num;
}

void outputVec(const Vec &vec, std::ostream &ost /* = std::cout*/)
{
  for (const auto &num : vec)
    ost << num << ' ';
  ost << std::endl;
}
} // namespace fft

#define PRINT_FFT(inp, ans, func) testPrintFFT(inp, ans, func, #func)

int inputOutput()
{
  using namespace fft;

  Vec inp, ans;

  inputVec(inp);
  inputVec(ans);
  auto N = inp.size();

  if (inp.size() != ans.size() || N <= 1)
  {
    std::cerr << "Input and answer must have same positive sizes" << std::endl;
    return 1;
  }
  if ((N & (N - 1)) != 0)
  {
    std::cerr << "Size must be a power of 2" << std::endl;
    return 1;
  }

  std::cout << "Vec size: " << N << std::endl;

  if (N <= 2048)
    PRINT_FFT(inp, ans, naiveFFT);
  PRINT_FFT(inp, ans, ctFFT);
  PRINT_FFT(inp, ans, ctParFFT);

  return 0;
}

template <std::forward_iterator It>
void variate(It beg, It end)
{
  std::cout << "N,    TIME" << std::endl;
  std::for_each(beg, end, [](auto size) {
    fft::Vec inp = fft::genData(size);
    fft::Vec res;
    auto elapsed = fft::runFFT(inp, res, fft::ctParFFT);
    std::cout << size << ", " << elapsed << std::endl;
  });
}

int main()
{
#if defined(IN_OUT_MODE)
  return inputOutput();
#else
  // [)
  constexpr std::pair shift_range{12, 21};
  std::array<std::size_t, shift_range.second - shift_range.first> sizes{};

  std::size_t cur_shift = shift_range.first;
  std::generate(sizes.begin(), sizes.end(),
                [&cur_shift]() { return 1 << cur_shift++; });

  variate(sizes.begin(), sizes.end());

#endif
}
