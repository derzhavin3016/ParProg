#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <gmp.h>

typedef unsigned uint;
typedef long double ldbl;

mpf_t fact( uint N );

void calc_sum(int rank, uint N, int commsize);

ldbl finalize( uint commsize );


int main( int argc, char *argv[] )
{
  int rank = 0, commsize = 0;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);

  if (argc >= 2)
  {
    if (commsize < 2)
    {
      printf("Commsize should be at least 2, given %i\n", commsize);
      return -1;
    }

    if (rank == 0)
      printf("e = %.*Lf\n", LDBL_DIG - 1, finalize(commsize));
    else
      calc_sum(rank, atoi(argv[1]), commsize);
  }
  else
    printf("USAGE: %s AMOUNT_OF_TERMS\n", argv[0]);

  MPI_Finalize();
}

ldbl finalize( uint commsize )
{
  ldbl result = 0.0;
  for (uint i = 1; i < commsize; ++i)
  {
    ldbl i_th_res = 0.0;
    MPI_Recv(&i_th_res, 1, MPI_LONG_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    result += i_th_res;
  }

  return result;
}

void calc_sum(int rank, uint N, int commsize)
{
  mpf_t part;
  mpz_set_d(part, 0);

  uint work_amount = N / (commsize - 1);
  uint start_i = work_amount * (rank - 1);
  
  if (rank == commsize - 1)
    work_amount += N % (commsize - 1);

  mpf_t term_i;
  mpf_set_d(term_i, 1.0);
  mpf_div(term_i, term_i, fact(start_i));

  for (uint i = start_i, end_i = start_i + work_amount; i < end_i;)
  {
    part += term_i;
    term_i /= ++i;
  }


  MPI_Send(&part, 1, MPI_LONG_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

mpf_t fact( uint N )
{
  mpf_t res;
  mpf_init_set_ui(res, 0);

  for (uint i = 1; i <= N; ++i)
    mpf_mul_ui(res, res, i);
  return res;
}