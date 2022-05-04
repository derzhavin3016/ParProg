#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <mpi.h>


using ldbl = long double;

ldbl f(ldbl x, ldbl t)
{
  return t * x;
}

ldbl phi(ldbl x)
{
  return x * x * x / 12;
}

ldbl psi(ldbl t)
{
  return t * t * t / 12;
}

int main(int argc, char *argv[])
{
  constexpr ldbl a = 1;
  constexpr ldbl h = 1e-3;
  constexpr ldbl tau = 1e-3;
  constexpr ldbl T = 10;
  constexpr ldbl X = 10;
  constexpr std::uint64_t Nt = T / tau + 1;
  constexpr std::uint64_t Nx = X / h + 1;

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
    for (std::uint64_t i = 0; i < Nx; ++i)
    {
      auto &cur_u = matr[row_id][i];
      if (row_id == 0)
        cur_u = phi(i * h);
      else
      {
        std::uint64_t src_id = row_id % commsize;
        MPI::COMM_WORLD.Recv(matr[row_id - 1][i], 1, MPI::LONG_DOUBLE, static_cast<int>(src_id ? src_id - 1 : commsize - 1), static_cast<int>(i));

        auto f_val = f((i + 0.5) * h, (row_id + 0.5) * tau);
        cur_u = 
      }

      MPI::COMM_WORLD.Send(&(cur_u), 1, MPI::LONG_DOUBLE, (row_id + 1) % commsize, static_cast<int>(i));
    }

  MPI::Finalize();
  return 0;
}