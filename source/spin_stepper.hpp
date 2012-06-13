#ifndef SPIN_STEPPER_HPP
#define SPIN_STEPPER_HPP

#include <thrust/device_vector.h>

#include "functors.hpp"

template< class VectorType , class ValueType >
class spin_stepper
{
public:

    typedef VectorType vector_type;
    typedef ValueType value_type;
    typedef typename vector_type::iterator iter_type;
    typedef typename vector_type::const_iterator const_iter_type;

    spin_stepper( int N , 
                  const vector_type &h_x , 
                  const vector_type &h_y , 
                  const vector_type &h_z  )
        : m_N( N ) , m_h_x( h_x ) , m_h_y( h_y ) , m_h_z( h_z )
    { }

    void do_step( vector_type &s_x , vector_type &s_y , vector_type &s_z , 
                  vector_type &b_x , vector_type &b_z , vector_type &b_y ,
                  vector_type &b_norm , const double dt )
    {
        using namespace thrust;

        /* compute /vec b and the norm B */
        thrust::for_each( 
            make_zip_iterator( make_tuple( 
                 make_zip_iterator( make_tuple (
                     s_x.begin() ,        // s_x[i-1]
                     s_x.begin()+2 ) ) ,  // s_x[i+1]
                 make_zip_iterator( make_tuple (
                     s_y.begin() ,
                     s_y.begin()+2 ) ),
                 make_zip_iterator( make_tuple (
                     s_z.begin() ,
                     s_z.begin()+2 ) ),
                 m_h_x.begin() ,
                 m_h_y.begin() ,
                 m_h_z.begin() ,
                 b_x.begin() ,
                 b_y.begin() ,
                 b_z.begin() ,
                 b_norm.begin() ) ) ,
            make_zip_iterator( make_tuple( 
                 make_zip_iterator( make_tuple (
                     s_x.end()-2 ,
                     s_x.end() ) ) ,
                 make_zip_iterator( make_tuple (
                     s_y.end()-2 ,
                     s_y.end() ) ),
                 make_zip_iterator( make_tuple (
                     s_z.end()-2 ,
                     s_z.end() ) ) ,
                 m_h_x.end() ,
                 m_h_y.end() ,
                 m_h_z.end() ,
                 b_x.end() ,
                 b_y.end() ,
                 b_z.end() ,
                 b_norm.end() ) ) ,
            b_functor< value_type >() );

        /* compute /vec s */
        thrust::for_each(
            make_zip_iterator( make_tuple(
                s_x.begin() ,
                s_y.begin() ,
                s_z.begin() ,
                b_x.begin() ,
                b_y.begin() ,
                b_z.begin() ,
                b_norm.begin() ) ) ,
            make_zip_iterator( make_tuple(
                s_x.end() ,
                s_y.end() ,
                s_z.end() ,
                b_x.end() ,
                b_y.end() ,
                b_z.end() ,
                b_norm.end() ) ) ,
            s_functor< value_type >( dt ) );
            
                
    }

private:
    int m_N;
    const vector_type &m_h_x;
    const vector_type &m_h_y;
    const vector_type &m_h_z; 
};

#endif
