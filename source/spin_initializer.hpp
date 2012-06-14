#ifndef SPIN_INITIALIZER_HPP
#define SPIN_INITIALIZER_HPP

#include <boost/random.hpp>
#include <boost/math/constants/constants.hpp>

template< typename ValueType >
class spin_initializer
{
public:
    typedef ValueType value_type;

    spin_initializer( value_type q , value_type nu , boost::mt19937 &rng )
        : m_q( q ) , m_nu( nu ) , m_rng( rng )
    {  }

    template< class VectorType >
    void init_normalized_random( VectorType &x , VectorType &y , VectorType &z )
    {
        for( int n=0 ; n<x.size() ; ++n )
        {
            x[n] = m_uniform_dist( m_rng );// - 0.5;
            y[n] = m_uniform_dist( m_rng );// - 0.5;
            z[n] = m_uniform_dist( m_rng );// - 0.5;
            value_type nrm = sqrt( x[n]*x[n] + y[n]*y[n] + z[n]*z[n] );
            x[n] /= nrm;
            y[n] /= nrm;
            z[n] /= nrm;
        }
    }

    template< class VectorType >
    void relax( const VectorType &h_x , const VectorType &h_y , const VectorType &h_z ,
                VectorType &s_x , VectorType &s_y , VectorType &s_z ,
                const double beta0 )
    {
        const int N = h_x.size();
        // relax every second spin:
        for( int n=0 ; n<N ; n+=2 )
        {
            // s have extra boundary condition spins
            value_type b_x = h_x[n] + s_x[n] + s_x[n+2];
            value_type b_y = h_y[n] + s_y[n] + s_y[n+2];
            value_type b_z = h_z[n] + s_z[n] + s_z[n+2];
            const value_type b = sqrt( b_x*b_x + b_y*b_y + b_z*b_z );
            b_x /= b; b_y /= b; b_z /= b;
            
            /* *******************
               TO BE FIXED
               ******************* */

            const value_type beta = beta0 + m_nu*cos( (n*m_q) / N );
            
            // what exactly should be lambda for the exp distr
            boost::exponential_distribution< value_type > exp_dist( b*beta );
            boost::variate_generator< boost::mt19937 & ,
                                      boost::exponential_distribution< value_type > > var_exp( m_rng , exp_dist );
            value_type cos_theta = var_exp( );

            // how to get a vector not perpendicular to b ?
            s_x[n+1] = cos_theta*b_x;
            s_y[n+1] = cos_theta*b_y;
            s_z[n+1] = cos_theta*b_z;
        }
    }

private:
    const value_type m_q;
    const value_type m_nu;
    boost::mt19937 &m_rng;
    boost::uniform_01< value_type > m_uniform_dist;
};

#endif
