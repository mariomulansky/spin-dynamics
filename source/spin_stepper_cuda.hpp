#ifndef SPIN_STEPPER_CUDA_HPP
#define SPIN_STEPPER_CUDA_HPP

#include <thrust/device_vector.h>

__global__ void timestep_kernel( double* s_x , double* s_y , double* s_z , double* h_x , double* h_y , double* h_z , double dt )
{
    const int i_even = 2*threadIdx.x;
    const int i_odd = 2*threadIdx.x+1;

    double b_x , b_y , b_z , b_norm , bs , co , si;

    // first even spins

    // start with calculating local b field
    b_x = h_x[i_even] + s_x[i_even-1] + s_x[i_even+1];
    b_y = h_y[i_even] + s_y[i_even-1] + s_z[i_even+1];
    b_z = h_z[i_even] + s_y[i_even-1] + s_z[i_even+1];
    b_norm = sqrt(b_x*b_x + b_y*b_y + b_z*b_z);
    b_x /= b_norm;
    b_y /= b_norm;
    b_z /= b_norm;

    // do spin rotation
    bs = s_x[i_even]*b_x + s_y[i_even]*b_y + s_z[i_even]*b_z;
    co = cos( dt*b_norm );
    si = sin( dt*b_norm );

    s_x[i_even] = b_x * bs + (s_x[i_even] - b_x*bs) * co + (b_y*s_z[i_even] - b_z*s_y[i_even]) * si;
    s_y[i_even] = b_y * bs + (s_y[i_even] - b_y*bs) * co + (b_z*s_x[i_even] - b_x*s_z[i_even]) * si;
    s_z[i_even] = b_z * bs + (s_z[i_even] - b_z*bs) * co + (b_x*s_y[i_even] - b_y*s_x[i_even]) * si;

    __syncthreads(); // make sure all even spins are updated

    // do the same for odd spins

    // start with calculating local b field
    b_x = h_x[i_odd] + s_x[i_odd-1] + s_x[i_odd+1];
    b_y = h_y[i_odd] + s_y[i_odd-1] + s_z[i_odd+1];
    b_z = h_z[i_odd] + s_y[i_odd-1] + s_z[i_odd+1];
    b_norm = sqrt(b_x*b_x + b_y*b_y + b_z*b_z);
    b_x/=b_norm;
    b_y/=b_norm;
    b_z/=b_norm;

    // do spin rotation
    bs = s_x[i_odd]*b_x + s_y[i_odd]*b_y + s_z[i_odd]*b_z;
    co = cos( dt*b_norm );
    si = sin( dt*b_norm );

    s_x[i_odd] = b_x * bs + (s_x[i_odd] - b_x*bs) * co + (b_y*s_z[i_odd] - b_z*s_y[i_odd]) * si;
    s_y[i_odd] = b_y * bs + (s_y[i_odd] - b_y*bs) * co + (b_z*s_x[i_odd] - b_x*s_z[i_odd]) * si;
    s_z[i_odd] = b_z * bs + (s_z[i_odd] - b_z*bs) * co + (b_x*s_y[i_odd] - b_y*s_x[i_odd]) * si;

    // finished
}

template< class VectorType , class ValueType >
class spin_stepper_cuda
{
public:

    typedef VectorType vector_type;
    typedef ValueType value_type;
    typedef typename vector_type::iterator iter_type;
    typedef typename vector_type::const_iterator const_iter_type;

    spin_stepper_cuda( int N , 
                       vector_type &h_x , 
                       vector_type &h_y , 
                       vector_type &h_z  )
        : m_N( N ) , m_h_x( h_x ) , m_h_y( h_y ) , m_h_z( h_z )
    { }


    void do_step( vector_type &s_x , vector_type &s_y , vector_type &s_z ,
                  const double dt )
    {
        // elements 0 and N+1 are for constant boundary condition
        value_type *s_x_ptr = thrust::raw_pointer_cast(s_x.data()+1);
        value_type *s_y_ptr = thrust::raw_pointer_cast(s_y.data()+1);
        value_type *s_z_ptr = thrust::raw_pointer_cast(s_z.data()+1);

        value_type *h_x_ptr = thrust::raw_pointer_cast(m_h_x.data());
        value_type *h_y_ptr = thrust::raw_pointer_cast(m_h_y.data());
        value_type *h_z_ptr = thrust::raw_pointer_cast(m_h_z.data());

        timestep_kernel<<<1,m_N/2>>>( s_x_ptr , s_y_ptr , s_z_ptr , h_x_ptr , h_y_ptr , h_z_ptr , dt );
    }


    void energies( vector_type &s_x , vector_type &s_y , vector_type &s_z ,
                   vector_type &energy ) 
    {
        //calculate energies and store result in b_norm
        thrust::for_each(
            thrust::make_zip_iterator( thrust::make_tuple( 
                 thrust::make_zip_iterator( thrust::make_tuple (
                     s_x.begin() ,        // s_x[i-1]
                     s_x.begin()+1 ,      // s_x[i]
                     s_x.begin()+2 ) ) ,  // s_x[i+1]
                 thrust::make_zip_iterator( thrust::make_tuple (
                     s_y.begin() ,
                     s_y.begin()+1 ,
                     s_y.begin()+2 ) ),
                 thrust::make_zip_iterator( thrust::make_tuple (
                     s_z.begin() ,
                     s_z.begin()+1 ,
                     s_z.begin()+2 ) ),
                 m_h_x.begin() ,
                 m_h_y.begin() ,
                 m_h_z.begin() ,
                 energy.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( 
                 thrust::make_zip_iterator( thrust::make_tuple (
                     s_x.end()-2 ,
                     s_x.end()-1 ,
                     s_x.end() ) ) ,
                 thrust::make_zip_iterator( thrust::make_tuple (
                     s_y.end()-2 ,
                     s_y.end()-1 ,
                     s_y.end() ) ),
                 thrust::make_zip_iterator( thrust::make_tuple (
                     s_z.end()-2 ,
                     s_z.end()-1 ,
                     s_z.end() ) ) ,
                 m_h_x.end() ,
                 m_h_y.end() ,
                 m_h_z.end() ,
                 energy.end() ) ) ,
            energy_functor< value_type >() );
    }

    int m_N;
    vector_type &m_h_x;
    vector_type &m_h_y;
    vector_type &m_h_z; 
};

#endif
