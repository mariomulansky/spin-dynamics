#include <iostream>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>

#include "spin_relax.hpp"

static const int N = 1000;
static const int steps = 51;
static const double dt = 0.1;
static const double beta0 = 0.0001;
static const double nu = 0.5;
static const double k_max = 1;

int main(int argc, char *argv[]) {

    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);

    printf("Process %d on %s out of %d\n", rank, processor_name, numprocs);

    const int cuda_device = rank % 2;
    cudaSetDevice( cuda_device );

    for( double J = 0.7 ; J > 0.1 ; J /= 1.1220184543 )
    {
        for( int k = 0 ; k <= k_max ; k++ )
        {
            if( k % numprocs == rank )
            {
                const double q = 2.0 * M_PI * N * (k+1)/(20*(k_max+1));
                spin_relax( N , J , q , steps , dt , beta0 , nu );
            }
        }
    }

    MPI_Finalize();
}
