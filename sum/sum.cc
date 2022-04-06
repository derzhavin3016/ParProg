#include <mpi.h>
#include <cstdio>
#include <iostream>

using uint = unsigned;
using ldbl = long double;

void calc_sum(int rank, uint N, int commsize)
{
  ldbl part = 0.0;

  uint work_amount = N / commsize;
  uint start_i = work_amount * (rank - 1) + 1;
  
  if (rank == commsize)
    work_amount += N % commsize;

  for (uint i = start_i, end_i = start_i + work_amount; i < end_i; ++i)
    part += 1.0 / i;

  MPI::COMM_WORLD.Send(&part, 1, MPI::LONG_DOUBLE, 0, 0);
}

ldbl finalize( uint commsize )
{
  ldbl result = 0.0;
  for (uint i = 1; i < commsize; ++i)
  {
    ldbl i_th_res = 0.0;
    MPI::COMM_WORLD.Recv(&i_th_res, 1, MPI::LONG_DOUBLE, i, 0);

    result += i_th_res;
  }

  return result;
}

int main( int argc, char *argv[] )
{
  MPI::Init(argc, argv);
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  if (argc >= 2)
  {
    if (rank == 0)
      printf("sum = %Lf\n", finalize(commsize));
    else
      calc_sum(rank, atoi(argv[1]), commsize - 1);
  }
  else if (rank == 0)
    printf("USAGE: %s AMOUNT_OF_SUM_ELEMS\n", argv[0]);

  MPI::Finalize();
}