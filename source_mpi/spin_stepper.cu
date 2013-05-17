#include <thrust/device_vector.h>

#include "spin_stepper.hpp"

#include "functors.hpp"
#include "strided_range.hpp"

template< class VectorType , class ValueType >
void spin_stepper<VectorType,ValueType>::do_step( vector_type &s_x , 
                                                  vector_type &s_y , 
                                                  vector_type &s_z ,
                                                  const double dt )
{
    // even spins
    compute_b( s_x , s_y , s_z , true );
    evolve_s(  s_x , s_y , s_z , dt , true );
    
    // odd spins
    compute_b( s_x , s_y , s_z , false );
    evolve_s(  s_x , s_y , s_z , dt , false );
    
}

template< class VectorType , class ValueType >
void spin_stepper<VectorType,ValueType>::energies( vector_type &s_x , 
                                                   vector_type &s_y , 
                                                   vector_type &s_z ,
                                                   vector_type &energy ) 
{
    //calculate energies and store result in b_norm
    thrust::for_each(
        thrust::make_zip_iterator( 
            thrust::make_tuple( 
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
        thrust::make_zip_iterator( 
            thrust::make_tuple( 
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

template< class VectorType , class ValueType >
void spin_stepper<VectorType,ValueType>::compute_b( vector_type &s_x , 
                                                    vector_type &s_y ,
                                                    vector_type &s_z , 
                                                    const bool even )
{
    int m;
    if( even ) m = 0;
    else m = 1;

    // make strided ranges (vector masks )
    typedef thrust::strided_range< typename vector_type::iterator > range_type;

    range_type rs_x_left( s_x.begin() + m , s_x.end()-3 + m , 2 );
    range_type rs_x_right( s_x.begin()+2 + m , s_x.end()-1 + m , 2 );
        
    range_type rs_y_left( s_y.begin() + m , s_y.end()-3 + m , 2 );
    range_type rs_y_right( s_y.begin()+2 + m , s_y.end()-1 + m , 2 );
        
    range_type rs_z_left( s_z.begin() + m , s_z.end()-3 + m , 2 );
    range_type rs_z_right( s_z.begin()+2 + m , s_z.end()-1 + m , 2 );
        
    range_type rh_x( m_h_x.begin() + m , m_h_x.end()-1 + m , 2 );
    range_type rh_y( m_h_y.begin() + m , m_h_y.end()-1 + m , 2 );
    range_type rh_z( m_h_z.begin() + m , m_h_z.end()-1 + m , 2 );

    range_type rb_x( m_b_x.begin() + m , m_b_x.end()-1 + m , 2 );
    range_type rb_y( m_b_y.begin() + m , m_b_y.end()-1 + m , 2 );
    range_type rb_z( m_b_z.begin() + m , m_b_z.end()-1 + m , 2 );

    range_type rb_norm( m_b_norm.begin() + m , m_b_norm.end()-1 + m , 2 );

/* compute /vec b and the norm B */
    thrust::for_each( 
        thrust::make_zip_iterator( thrust::make_tuple( 
                                       thrust::make_zip_iterator( thrust::make_tuple (
                                                                      rs_x_left.begin() ,       // s_x[2i-1]
                                                                      rs_x_right.begin() ) ) ,  // s_x[2i+1]
                                       thrust::make_zip_iterator( thrust::make_tuple (
                                                                      rs_y_left.begin() ,
                                                                      rs_y_right.begin() ) ) ,
                                       thrust::make_zip_iterator( thrust::make_tuple (
                                                                      rs_z_left.begin() ,
                                                                      rs_z_right.begin() ) ) ,// s_z[2i+1] ) ),
                                       rh_x.begin() ,
                                       rh_y.begin() ,
                                       rh_z.begin() ,
                                       m_b_x.begin() ,
                                       m_b_y.begin() ,
                                       m_b_z.begin() ,
                                       m_b_norm.begin() ) ) ,
        thrust::make_zip_iterator( thrust::make_tuple( 
                                       thrust::make_zip_iterator( thrust::make_tuple (
                                                                      rs_x_left.end() ,   // s_x[2i-1]
                                                                      rs_x_right.end() ) ) ,  // s_x[2i+1]
                                       thrust::make_zip_iterator( thrust::make_tuple (
                                                                      rs_y_left.end() ,
                                                                      rs_y_right.end() ) ) ,
                                       thrust::make_zip_iterator( thrust::make_tuple (
                                                                      rs_z_left.end() ,
                                                                      rs_z_right.end() ) ) ,// s_z[2i+1] ) ),
                                       rh_x.end() ,
                                       rh_y.end() ,
                                       rh_z.end() ,
                                       m_b_x.end() ,
                                       m_b_y.end() ,
                                       m_b_z.end() ,
                                       m_b_norm.end() ) ) ,            
        b_functor< value_type >() );        
}

template< class VectorType , class ValueType >
void spin_stepper<VectorType,ValueType>::evolve_s( vector_type &s_x , 
                                                   vector_type &s_y , 
                                                   vector_type &s_z , 
                                                   const double dt ,
                                                   const bool even )
{
    int m;
    if( even ) m = 0;
    else m = 1;

    // make strided ranges (vector masks )
    typedef thrust::strided_range< typename vector_type::iterator > range_type;

    range_type rs_x( s_x.begin()+1 + m , s_x.end()-2 + m , 2 );
    range_type rs_y( s_y.begin()+1 + m , s_y.end()-2 + m , 2 );
    range_type rs_z( s_z.begin()+1 + m , s_z.end()-2 + m , 2 );

    range_type rb_x( m_b_x.begin() + m , m_b_x.end()-1 + m , 2 );
    range_type rb_y( m_b_y.begin() + m , m_b_y.end()-1 + m , 2 );
    range_type rb_z( m_b_z.begin() + m , m_b_z.end()-1 + m , 2 );

    range_type rb_norm( m_b_norm.begin() + m , m_b_norm.end()-1 + m , 2 );
    /* compute /vec s */
    thrust::for_each(
        thrust::make_zip_iterator( thrust::make_tuple(
                                       rs_x.begin() ,
                                       rs_y.begin() ,
                                       rs_z.begin() ,
                                       m_b_x.begin() ,
                                       m_b_y.begin() ,
                                       m_b_z.begin() ,
                                       m_b_norm.begin() ) ) ,
        thrust::make_zip_iterator( thrust::make_tuple(
                                       rs_x.end() ,
                                       rs_y.end() ,
                                       rs_z.end() ,
                                       m_b_x.end() ,
                                       m_b_y.end() ,
                                       m_b_z.end() ,
                                       m_b_norm.end() ) ) ,
        s_functor< value_type >( dt ) );
}

typedef thrust::device_vector<double> vector_type;

void foo()
{
    vector_type x,y,z;
    spin_stepper<vector_type,double> stepper( 0 , x , y, z );
}
