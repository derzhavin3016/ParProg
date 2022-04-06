#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned uint;
typedef long double ldbl;

void print_status(uint msg, int rank)
{
  printf("===\nRank = %d\nCurrent msg = %d\n===\n", rank, msg);
}

void do_zero(uint msg, uint commsize)
{
  print_status(msg, 0);
  uint dst = 1, from = commsize - 1;
  if (commsize == 1)
    dst = from = 0;

  MPI_Send(&msg, 1, MPI_UNSIGNED, dst, 0, MPI_COMM_WORLD);

  MPI_Recv(&msg, 1, MPI_UNSIGNED, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  print_status(++msg, 0);
}

void ring_msg(uint msg, int rank, uint commsize)
{
  if (rank == 0)
  {
    do_zero(msg, commsize);
    return;
  }

  uint from_rank = rank - 1;
  uint to_rank = (uint)rank == commsize - 1 ? 0 : rank + 1;
  uint cur_msg = 0;
  MPI_Recv(&cur_msg, 1, MPI_UNSIGNED, from_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  print_status(++cur_msg, rank);

  MPI_Send(&cur_msg, 1, MPI_UNSIGNED, to_rank, 0, MPI_COMM_WORLD);
}

int main( int argc, char *argv[] )
{
  int rank = 0, commsize = 0;
  MPI_Init(&argc, &argv);
  setbuf(stdout, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);

  ring_msg(0, rank, commsize);

  MPI_Finalize();
}