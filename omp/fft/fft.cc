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

int main()
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
}
