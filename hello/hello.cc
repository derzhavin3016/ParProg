#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  MPI::Init(argc, argv);
  printf("Hello World\nCommunicator size = %d My rank = %d\n", MPI::COMM_WORLD.Get_size(), MPI::COMM_WORLD.Get_rank());
  MPI::Finalize();
}