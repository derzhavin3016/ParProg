// OMP 1д task
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string_view>

#include "timer.hh"

constexpr std::size_t ISIZE = 1000;
constexpr std::size_t JSIZE = 1000;

using ArrTy = std::array<std::array<double, JSIZE>, ISIZE>;

void printArr(const ArrTy &arr, std::ostream &ost)
{
  for (auto &&row : arr)
  {
    for (auto elem : row)
      ost << elem << " ";
    ost << std::endl;
  }
}

using ProcFunc = std::function<void(ArrTy &)>;

void processArr(ArrTy &arr)
{
  for (std::size_t i = 0; i < ISIZE - 1; i++)
    for (std::size_t j = 6; j < JSIZE; j++)
      arr[i][j] = std::sin(0.2 * arr[i + 1][j - 6]);
}

void processArrPar(ArrTy &arr)
{}

void measureDump(ProcFunc f, ArrTy &arr, std::string_view filename)
{
  timer::Timer tim;
  f(arr);
  auto elapsed_mcs = tim.elapsed_mcs();

  std::cout << "Elapsed time " << elapsed_mcs / 1000.0l << " ms" << std::endl;

  std::ofstream of(filename.data());
  if (!of)
  {
    std::cerr << "Cannot open file for results" << std::endl;
    return;
  }
  printArr(arr, of);
}

void initArr(ArrTy &a)
{
  // Fill array with data
  for (std::size_t i = 0; i < a.size(); i++)
  {
    auto &ai = a[i];
    for (std::size_t j = 0; j < ai.size(); j++)
    {
      ai[j] = 10 * i + j;
    }
  }
}

int main()
{
  ArrTy a{};
  initArr(a);

#if defined(SEQ_VER)
  std::cout << "Sequential:" << std::endl;
  measureDump(processArr, a, "seq.txt");
#else
  std::cout << "Parallel:" << std::endl;
  measureDump(processArrPar, a, "par.txt");
#endif
}
