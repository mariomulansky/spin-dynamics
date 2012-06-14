#ifndef FUNCTORS_HPP
#define FUNCTORS_HPP

#include <cmath>


template< typename Value >
struct b_functor
{
    typedef Value value_type;

    template< class Tuple >
    //__host__ __device__
    void operator()( Tuple t ) const
    {
        // extract the values from the tuple
        const value_type s_x_l = thrust::get< 0 >( thrust::get< 0 >( t ) );
        const value_type s_x_r = thrust::get< 1 >( thrust::get< 0 >( t ) );

        const value_type s_y_l = thrust::get< 0 >( thrust::get< 1 >( t ) );
        const value_type s_y_r = thrust::get< 1 >( thrust::get< 1 >( t ) );

        const value_type s_z_l = thrust::get< 0 >( thrust::get< 2 >( t ) );
        const value_type s_z_r = thrust::get< 1 >( thrust::get< 2 >( t ) );

        const value_type h_x = thrust::get< 3 >( t );
        const value_type h_y = thrust::get< 4 >( t );
        const value_type h_z = thrust::get< 5 >( t );

        thrust::get< 6 >( t ) = h_x + s_x_l + s_x_r; // b_x
        thrust::get< 7 >( t ) = h_y + s_y_l + s_y_r; // b_y
        thrust::get< 8 >( t ) = h_z + s_z_l + s_z_r; // b_z
        
        // b_norm:
        thrust::get< 9 >( t ) = sqrt( thrust::get< 6 >( t )*thrust::get< 6 >( t ) + 
                                      thrust::get< 7 >( t )*thrust::get< 7 >( t ) + 
                                      thrust::get< 8 >( t )*thrust::get< 8 >( t ) );
        if( thrust::get< 9 >( t ) > 0.0 )
        {
            thrust::get< 6 >( t ) /= thrust::get< 9 >( t );
            thrust::get< 7 >( t ) /= thrust::get< 9 >( t );
            thrust::get< 8 >( t ) /= thrust::get< 9 >( t );
        }

        //std::clog << thrust::get< 9 >( t ) << std::endl;
    }
};

template< typename Value >
struct s_functor
{
    typedef Value value_type;

    s_functor( const value_type dt )
        : m_dt( dt )
    { }

    template< class Tuple >
    __host__ __device__
    void operator()( Tuple t ) const
    {
        using std::cos;
        using std::sin;
        // extract values

        const value_type s_x = thrust::get< 0 >( t );
        const value_type s_y = thrust::get< 1 >( t );
        const value_type s_z = thrust::get< 2 >( t );

        const value_type b_x = thrust::get< 3 >( t );
        const value_type b_y = thrust::get< 4 >( t );
        const value_type b_z = thrust::get< 5 >( t );

        const value_type bs = s_x*b_x + s_y*b_y + s_z*b_z;

        const value_type co = cos( m_dt * thrust::get< 6 >( t ) );
        const value_type si = sin( m_dt * thrust::get< 6 >( t ) );

        thrust::get< 0 >( t ) = b_x * bs + (s_x - b_x*bs) * co + (b_y*s_z - b_z*s_y) * si;
        thrust::get< 1 >( t ) = b_y * bs + (s_y - b_y*bs) * co + (b_z*s_x - b_x*s_z) * si;
        thrust::get< 2 >( t ) = b_z * bs + (s_z - b_z*bs) * co + (b_x*s_y - b_y*s_x) * si;        
        
    }

    value_type m_dt;
};


template< typename ValueType >
struct energy_functor
{
    typedef ValueType value_type;

    template< class Tuple >
    __host__ __device__
    void operator()( Tuple t ) const
    {
        // extract the values from the tuple
        const value_type s_x_l = thrust::get< 0 >( thrust::get< 0 >( t ) );
        const value_type s_x   = thrust::get< 1 >( thrust::get< 0 >( t ) );
        const value_type s_x_r = thrust::get< 2 >( thrust::get< 0 >( t ) );

        const value_type s_y_l = thrust::get< 0 >( thrust::get< 1 >( t ) );
        const value_type s_y   = thrust::get< 1 >( thrust::get< 1 >( t ) );
        const value_type s_y_r = thrust::get< 2 >( thrust::get< 1 >( t ) );

        const value_type s_z_l = thrust::get< 0 >( thrust::get< 2 >( t ) );
        const value_type s_z   = thrust::get< 1 >( thrust::get< 2 >( t ) );
        const value_type s_z_r = thrust::get< 2 >( thrust::get< 2 >( t ) );

        const value_type h_x = thrust::get< 3 >( t );
        const value_type h_y = thrust::get< 4 >( t );
        const value_type h_z = thrust::get< 5 >( t );

        const value_type h = static_cast< value_type >( 0.5 );
        thrust::get< 6 >( t )  = ( h_x + h*s_x_l + h*s_x_r ) * s_x;
        thrust::get< 6 >( t ) += ( h_y + h*s_y_l + h*s_y_r ) * s_y;
        thrust::get< 6 >( t ) += ( h_z + h*s_z_l + h*s_z_r ) * s_z;
    }
};

#endif
