#ifndef FOURIER_ANALYZER_CUDA
#define FOURIER_ANALYZER_CUDA

#include <iostream>
#include <vector>
#include <math.h>

#include <thrust/device_vector.h>

__global__ void fourier_kernel( const double* s_x , const double* s_y , const double* s_z , 
                                double* h_x , double* h_y , double* h_z ,
                                double* co , double* si , double* res_r , double* res_i )
{
    extern __shared__ double shared[];

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int BLOCK_SIZE = blockDim.x;

    double *real = &shared[0];
    double *imag = &shared[2*BLOCK_SIZE];

    // calculate local energy
    const double e = (h_x[i] + 0.5*s_x[i-1] + 0.5*s_x[i+1]) * s_x[i]
        + (h_y[i] + 0.5*s_y[i-1] + 0.5*s_y[i+1]) * s_y[i]
        + (h_z[i] + 0.5*s_z[i-1] + 0.5*s_z[i+1]) * s_z[i];

    // get the inner product with the sine and cosine
    real[threadIdx.x] = e*co[i];
    imag[threadIdx.x] = e*si[i];

    // make sure calculation is ready
    __syncthreads();

    int nTotalThreads = BLOCK_SIZE;
    int thread2;

    // now do the reduction using a binary tree
    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);	// divide by two
        // the first half of the threads will sum real
        // the second half imag
        thread2 = threadIdx.x + halfPoint;
        if (threadIdx.x < halfPoint)
        {
            real[threadIdx.x] += real[thread2];
        } else {
            imag[threadIdx.x] += imag[thread2];
        }
        __syncthreads();
        
        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }

    // finally store the results
    if( threadIdx.x == 0 )
    {
        res_r[blockIdx.x] = real[0];
        res_i[blockIdx.x] = imag[0];
    }
}
    
template< class VectorType , class ValueType >
class fourier_analyzer_cuda
{
public:
    
    typedef VectorType vector_type;
    typedef ValueType value_type;

    fourier_analyzer_cuda( const int N , const value_type q , 
                           vector_type &h_x , 
                           vector_type &h_y , 
                           vector_type &h_z ,
                           int block_size )
        : m_N( N ) , m_co( N ) , m_si( N ) , 
          m_h_x( h_x ) , m_h_y( h_y ) , m_h_z( h_z ) , m_block_size( block_size )
    { 
        std::vector< value_type > co( N );
        std::vector< value_type > si( N );
        for( int n=0 ; n<N ; ++n )
        {
            //using std::cos;
            //using std::sin;

            co[n] = cos( (q*n) / N );
            si[n] = sin( (q*n) / N );
        }
        thrust::copy( co.begin() , co.end() , m_co.begin() );
        thrust::copy( si.begin() , si.end() , m_si.begin() );

        cudaMalloc( (void**) &m_res_r_ptr , sizeof(value_type) );
        cudaMalloc( (void**) &m_res_i_ptr , sizeof(value_type) );
    }

    value_type analyze( const vector_type &s_x , const vector_type &s_y , const vector_type &s_z )
    {
        value_type real = 0.0;
        value_type imag = 0.0;

        // initialize result values to 0
        cudaMemcpy( m_res_r_ptr , &real_f , sizeof(value_type) , cudaMemcpyHostToDevice );
        cudaMemcpy( m_res_i_ptr , &imag_f , sizeof(value_type) , cudaMemcpyHostToDevice );

        //std::cout << "cudaMalloc ready" << std::endl;

        //leave out boundary values, actual data starts at index 1
        const value_type *s_x_ptr = thrust::raw_pointer_cast(s_x.data()+1);
        const value_type *s_y_ptr = thrust::raw_pointer_cast(s_y.data()+1);
        const value_type *s_z_ptr = thrust::raw_pointer_cast(s_z.data()+1);

        value_type *h_x_ptr = thrust::raw_pointer_cast(m_h_x.data());
        value_type *h_y_ptr = thrust::raw_pointer_cast(m_h_y.data());
        value_type *h_z_ptr = thrust::raw_pointer_cast(m_h_z.data());

        value_type *co_ptr = thrust::raw_pointer_cast(m_co.data());
        value_type *si_ptr = thrust::raw_pointer_cast(m_si.data());

        // the fourier component is calculated as float
        int shared_mem_size = m_block_size*sizeof(value_type);

        //std::cout << "starting kernel..." << std::endl;

        fourier_kernel<<< m_N/m_block_size , m_block_size , shared_mem_size >>>
            //fourier_kernel<<< m_N/m_block_size , m_block_size >>>
            ( s_x_ptr , s_y_ptr , s_z_ptr , 
              h_x_ptr , h_y_ptr , h_z_ptr , 
              co_ptr , si_ptr , m_res_r_ptr , m_res_i_ptr );

        cudaMemcpy( &real , m_res_r_ptr , sizeof(value_type) , cudaMemcpyDeviceToHost );
        cudaMemcpy( &imag , m_res_i_ptr , sizeof(value_type) , cudaMemcpyDeviceToHost );

        //std::cout << "copy finished..." << std::endl;
        
        return real*real + imag*imag;
    }

    ~fourier_analyzer_cuda()
    {
        cudaFree( m_res_r_ptr );
        cudaFree( m_res_i_ptr );
    }

private:
    int m_N;
    vector_type m_co;
    vector_type m_si;
    vector_type &m_h_x;
    vector_type &m_h_y;
    vector_type &m_h_z; 
    int m_block_size;
    value_type *m_res_r_ptr , *m_res_i_ptr;
};

#endif
