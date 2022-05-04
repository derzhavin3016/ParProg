#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

using uint = unsigned;
using ldbl = long double;

enum class DATA : int
{
  SUM,
  LTERM
};

void calc_sum(int rank, uint N, int commsize);

ldbl finalize(uint commsize);

int main(int argc, char *argv[])
{
  MPI::Init(argc, argv);

  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  if (argc >= 2)
  {
    if (rank == 0)
    {
      clock_t start = clock();
      ldbl res = finalize(commsize);
      clock_t end = clock();

      printf("e = %.*Lf\n", LDBL_DIG - 1, res);
      printf("M_E = %.*lf\n", DBL_DIG - 1, M_E);
      printf("|e - M_E| = %.*Lf\n", LDBL_DIG - 1, fabsl(res - M_E));

      printf("Executed in %lf s\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    }
    else
      calc_sum(rank, atoi(argv[1]), commsize - 1);
  }
  else if (rank == 0)
    printf("USAGE: %s NUM_OF_LAST_TERM\n", argv[0]);

  MPI::Finalize();
}

ldbl finalize(uint commsize)
{
  ldbl result = 1.0;
  ldbl cur_fact = 1.0;

  for (uint rank = 1; rank < commsize; ++rank)
  {
    ldbl cur_res = 0.0, last_term = 0.0;
    MPI::COMM_WORLD.Recv(&cur_res, 1, MPI::LONG_DOUBLE, rank, (int)DATA::SUM);
    MPI::COMM_WORLD.Recv(&last_term, 1, MPI::LONG_DOUBLE, rank, (int)DATA::LTERM);

    result += cur_res * cur_fact;
    cur_fact *= last_term;
  }

  return result;
}

void calc_sum(int rank, uint N, int commsize)
{
  ldbl part = 0.0;

  uint work_amount = N / commsize;
  uint start_i = work_amount * (rank - 1) + 1;

  if (rank == commsize)
    work_amount += N % commsize;

  ldbl term = 1.0;
  for (uint i = start_i, end_i = start_i + work_amount; i < end_i; ++i)
  {
    term /= i;
    part += term;
  }

  MPI::COMM_WORLD.Send(&part, 1, MPI::LONG_DOUBLE, 0, (int)DATA::SUM);
  MPI::COMM_WORLD.Send(&term, 1, MPI::LONG_DOUBLE, 0, (int)DATA::LTERM);
}
