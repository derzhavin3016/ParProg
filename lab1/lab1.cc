#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <mpi.h>

using ldbl = long double;

constexpr ldbl a = 1;
constexpr ldbl h = 1e-3;
constexpr ldbl tau = 1e-3;
constexpr ldbl T = 1;
constexpr ldbl X = 1;
constexpr std::uint64_t Nt = T / tau + 1;
constexpr std::uint64_t Nx = X / h + 1;
constexpr ldbl PI = 3.14159265358979323846;

ldbl f(ldbl x, ldbl t)
{
  return t * x;
}

ldbl phi(ldbl x)
{
  return std::cos(PI * x);
}

ldbl psi(ldbl t)
{
  return std::exp(-t);
}

void print_res(std::ostream &ost, const std::vector<ldbl> &res)
{
  ost << "tau: " << tau << std::endl;
  ost << "h: " << h << std::endl;
  ost << "T: " << T << std::endl;
  ost << "X: " << X << std::endl;
  ost << "Nx: " << Nx << std::endl;
  ost << "Nt: " << Nt << std::endl;

  for (std::size_t i = 0, sz = Nt * Nx; i < sz; ++i)
    ost << res[i] << std::endl;
}

int main(int argc, char *argv[])
{
  MPI::Init(argc, argv);

  std::vector<std::vector<ldbl>> matr;
  matr.resize(Nt);
  for (auto &row : matr)
    row.resize(Nx);

  // Fill initial values
  for (std::uint64_t i = 0; i < Nt; ++i)
    matr[i][0] = psi(i * tau);

  auto rank = static_cast<std::uint64_t>(MPI::COMM_WORLD.Get_rank());
  auto commsize = static_cast<std::uint64_t>(MPI::COMM_WORLD.Get_size());

  for (auto row_id = rank; row_id < Nt; row_id += commsize)
  {
    for (std::uint64_t i = 1; i < Nx; ++i)
    {
      if (row_id == 0)
        matr[row_id][i] = phi(i * h);
      else
      {
        MPI::COMM_WORLD.Recv(&(matr[row_id - 1][i]), 1, MPI::LONG_DOUBLE,
                             static_cast<int>(rank ? rank - 1 : commsize - 1),
                             static_cast<int>(i));

        auto f_val = f((i + 0.5) * h, (row_id - 0.5) * tau);
        matr[row_id][i] = matr[row_id - 1][i] + matr[row_id - 1][i - 1] - matr[row_id][i - 1] -
                a * tau / h * (-matr[row_id][i - 1] + matr[row_id - 1][i] - matr[row_id - 1][i - 1]) +
                2 * tau * f_val;
        matr[row_id][i] /= 1 + a * tau / h;
      }

      MPI::COMM_WORLD.Send(&(matr[row_id][i]), 1, MPI::LONG_DOUBLE,
                           (row_id + 1) % commsize, static_cast<int>(i));
    }
  }

  std::vector<ldbl> res;
  // Fill Nt in a way that commsize becomes it's factor
  auto Nt_filled = Nt + (commsize - Nt % commsize) % commsize;
  if (rank == 0) res.resize(Nt_filled * Nx);

  // Dummy vector for free processors
  std::vector<ldbl> empty(Nx, 0);

  for (auto row_id = rank; row_id < Nt_filled; row_id += commsize)
  {
    if (commsize > 1)
    {
      auto *src = row_id < Nt ? matr[row_id].data() : empty.data();

      MPI::COMM_WORLD.Gather(src, Nx, MPI::LONG_DOUBLE,
                             res.data() + row_id * Nx, Nx, MPI::LONG_DOUBLE, 0);
    }
    else if (row_id < Nt)
      std::copy(matr[row_id].begin(), matr[row_id].end(), res.begin() + row_id * Nx);
    else
      break;
  }


  if (rank == 0)
  {
    std::string name = argc > 1 ? argv[1] : "res.txt";
    std::ofstream f(name);
    print_res(f, res);
  }

  MPI::Finalize();
  return 0;
}