// OMP 1д task
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string_view>
#include <vector>

#include "timer.hh"

constexpr std::size_t ISIZE = 5000;
constexpr std::size_t JSIZE = 5000;

using ArrTy = std::vector<std::vector<double>>;

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
  // Original cycle
  // for (std::size_t i = 0; i < ISIZE - 1; i++)
  //   for (std::size_t j = 6; j < JSIZE; j++)
  //     arr[i][j] = std::sin(0.2 * arr[i + 1][j - 6]);
  // Normalized version
  for (std::size_t i = 0; i < ISIZE - 1; i++)
    for (std::size_t j = 0; j < JSIZE - 6; j++)
      arr[i][j + 6] = std::sin(0.2 * arr[i + 1][j]);
}
// Check bernstein condition:
// F(k1, k2) = (k1, k2 + 6)
// G(l1, l2) = (l1 + 1, l2)
// Have a system
// k1 = l1 + 1
// k2 + 6 = l2
// => l = (k1 - 1, k2 + 6)
// D = l - k = (-1, 6) => d = (>, <)
// > ==> i anti-dependency
// < ==> j true-dependency

void processArrPar(ArrTy &arr)
{
  for (std::size_t i = 0; i < ISIZE - 1; i++)
  {
#pragma omp parallel for schedule(static, 6)
    for (std::size_t j = 0; j < JSIZE - 6; j++)
      arr[i][j + 6] = std::sin(0.2 * arr[i + 1][j]);
  }
}

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
  a.resize(ISIZE);
  // Fill array with data
  for (std::size_t i = 0; i < a.size(); i++)
  {
    auto &ai = a[i];
    ai.resize(JSIZE);
    for (std::size_t j = 0; j < ai.size(); j++)
      ai[j] = 10 * i + j;
  }
}

int main()
{
  ArrTy a{};
  initArr(a);

  std::cout << "Sequential:" << std::endl;
  measureDump(processArr, a, "seq.txt");

  initArr(a);
  std::cout << "Parallel:" << std::endl;
  measureDump(processArrPar, a, "par.txt");
}
