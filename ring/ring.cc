#include <mpi.h>
#include <cstdio>
#include <cstdlib>

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

  MPI::COMM_WORLD.Send(&msg, 1, MPI::UNSIGNED, dst, 0);

  MPI::COMM_WORLD.Recv(&msg, 1, MPI::UNSIGNED, from, 0);
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
  MPI::COMM_WORLD.Recv(&cur_msg, 1, MPI::UNSIGNED, from_rank, 0);

  print_status(++cur_msg, rank);

  MPI::COMM_WORLD.Send(&cur_msg, 1, MPI::UNSIGNED, to_rank, 0);
}

int main( int argc, char *argv[] )
{
  MPI::Init(argc, argv);
  std::setbuf(stdout, nullptr);

  ring_msg(0, MPI::COMM_WORLD.Get_rank(), MPI::COMM_WORLD.Get_size());

  MPI::Finalize();
}