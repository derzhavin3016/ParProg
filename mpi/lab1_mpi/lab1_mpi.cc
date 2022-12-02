#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

constexpr std::size_t ISIZE = 5000;
constexpr std::size_t JSIZE = 5000;

using ArrTy = std::vector<std::vector<double>>;
using Dumper = std::function<void(const ArrTy &, std::ostream &)>;

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
  // for (std::size_t i = 1; i < ISIZE; i++)
  //   for (std::size_t j = 3; j < JSIZE - 1; j++)
  //     arr[i][j] = std::sin(2 * arr[i - 1][j - 3]);
  // Normalized version
  for (std::size_t i = 0; i < ISIZE - 1; i++)
    for (std::size_t j = 0; j < JSIZE - 4; j++)
      arr[i + 1][j + 3] = std::sin(2 * arr[i][j]);
}
// Check bernstein condition:
// F(k1, k2) = (k1 + 1, k2 + 3)
// G(l1, l2) = (l1, l2)
// Have a system
// k1 + 1 = l1
// k2 + 3 = l2
// => l = (k1 + 1, k2 + 3)
// D = l - k = (1, 6) => d = (<, <)
// < ==> i true-dependency
// < ==> j true-dependency

void processArrPar(ArrTy &arr)
{
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  for (std::size_t i = rank; i < JSIZE - 4; i += commsize)
    for (std::size_t j = 0; j < ISIZE - 1; j++)
      arr[i + 3][j + 1] = std::sin(2 * arr[i][j]);

  for (std::size_t i = rank; i < JSIZE - 4; i += commsize)
  {
    if (rank != 0)
    {
      MPI::COMM_WORLD.Send(arr[i].data() + 1, ISIZE - 1, MPI::DOUBLE, 0,
                           static_cast<int>(i));
      continue;
    }

    for (std::size_t id = 1; static_cast<int>(id) < commsize; ++id)
      if (i + id < JSIZE - 4)
        MPI::COMM_WORLD.Recv(arr[i + id].data() + 1, ISIZE - 1, MPI::DOUBLE, id,
                             static_cast<int>(i + id));
  }
}

void measureDump(ProcFunc f, ArrTy &arr, std::string_view filename,
                 Dumper dumper)
{
  auto rank = MPI::COMM_WORLD.Get_rank();
  MPI::COMM_WORLD.Barrier();
  auto tic = MPI::Wtime();
  f(arr);
  MPI::COMM_WORLD.Barrier();
  auto toc = MPI::Wtime();

  if (rank != 0)
  {
    arr.clear();
    return;
  }

  std::cout << "Elapsed time " << (toc - tic) * 1000 << " ms" << std::endl;

  std::ofstream of(filename.data());
  if (!of)
  {
    std::cerr << "Cannot open file for results" << std::endl;
    return;
  }
  dumper(arr, of);
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

void initTArr(ArrTy &a)
{
  a.resize(JSIZE);
  // Fill array with data
  for (std::size_t j = 0; j < a.size(); j++)
  {
    auto &aj = a[j];
    aj.resize(ISIZE);
    for (std::size_t i = 0; i < aj.size(); i++)
      aj[i] = 10 * i + j;
  }
}

void printTArr(const ArrTy &a, std::ostream &ost)
{
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
      ost << a[j][i] << " ";
    ost << std::endl;
  }
}

void doSeq()
{
  ArrTy a{};
  initArr(a);

  std::cout << "Sequential:" << std::endl;
  measureDump(processArr, a, "seq.txt", printArr);
}

void doPar()
{
  ArrTy a{};
  initTArr(a);

  if (MPI::COMM_WORLD.Get_rank() == 0)
    std::cout << "Parallel:" << std::endl;
  measureDump(processArrPar, a, "par.txt", printTArr);
}

int main(int argc, char *argv[])
{
  MPI::Init(argc, argv);
  auto commsz = MPI::COMM_WORLD.Get_size();

  if (commsz == 1)
    doSeq();
  else if (commsz == 3)
    doPar();
  else if (MPI::COMM_WORLD.Get_rank() == 0)
    std::cerr << "Commsize is not right " << commsz << " (required 3)"
              << std::endl;

  MPI::Finalize();
}
