// OMP 1д task
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

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

int main(void)
{
  ArrTy a{};
  std::size_t i = 0, j = 0;

  for (i = 0; i < ISIZE; i++)
  {
    for (j = 0; j < JSIZE; j++)
    {
      a[i][j] = 10 * i + j;
    }
  }
  for (i = 0; i < ISIZE - 1; i++)
  {
    for (j = 6; j < JSIZE; j++)
    {
      a[i][j] = std::sin(0.2 * a[i + 1][j - 6]);
    }
  }

  std::ofstream of("result.txt");
  if (!of)
  {
    std::cout << "Cannot open file for results" << std::endl;
    return 1;
  }
  printArr(a, of);
}
