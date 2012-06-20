#ifndef SPIN_INITIALIZER_HPP
#define SPIN_INITIALIZER_HPP

#include <cmath>
#include <cstdlib>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>


template< typename ValueType >
class spin_initializer
{
public:
    typedef ValueType value_type;

    spin_initializer( value_type q , value_type nu )
        : m_q( q ) , m_nu( nu )
    {  }

    template< class VectorType >
    void init_normalized_random( VectorType &x , VectorType &y , VectorType &z )
    {
        for( int n=0 ; n<x.size() ; ++n )
        {
            x[n] = static_cast< value_type >( rand() ) / RAND_MAX - 0.5; 
            //m_uniform_dist( m_rng ) - 0.5;
            y[n] = static_cast< value_type >( rand() ) / RAND_MAX - 0.5;
            z[n] = static_cast< value_type >( rand() ) / RAND_MAX - 0.5;
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
        using std::sin;
        using std::cos;
        using std::acos;
        using std::abs;
        const int N = h_x.size();
        // relax every second spin:
        std::ofstream bfile( "beta.dat" );
            
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
            
            value_type R = static_cast< value_type >( rand() ) / RAND_MAX;
            
            value_type cos_theta = 1.0/(b*beta) * log( (1.0-R)*exp(b*beta) + R*exp(-b*beta) );
            value_type phi = 2.0 * M_PI * static_cast< value_type >( rand() ) / RAND_MAX ;
            
            bfile << n << '\t' << beta << '\t' << cos_theta << std::endl;

            // s in coordinate system of b: e_z points into b
            boost::numeric::ublas::vector< value_type > s( 3 );
            s(0) = sqrt( 1.0 - cos_theta*cos_theta ) * cos( phi );
            s(1) = sqrt( 1.0 - cos_theta*cos_theta ) * sin( phi );
            s(2) = cos_theta;
            
            //std::clog << b << ", " << s(0)*s(0) + s(1)*s(1) + s(2)*s(2) << ", ";

            //std::clog << cos_theta;

            // rotate s into global coordinate system
            // rotation vector r = b \cross e_z
            value_type r_x = 1.0/sqrt( b_x*b_x + b_y*b_y ) * b_y;
            value_type r_y = - 1.0/sqrt( b_x*b_x + b_y*b_y ) * b_x;
            // rotation angle omega: 
            value_type omega = -acos( b_z );
            // rotation matrix:
            boost::numeric::ublas::matrix< value_type > rot( 3 , 3 );
            rot( 0 , 0 ) = cos( omega ) + r_x*r_x * ( 1.0 - cos( omega ) );
            rot( 0 , 1 ) = r_x*r_y * ( 1.0 - cos( omega ) );
            rot( 0 , 2 ) = r_y * sin( omega );
            
            rot( 1 , 0 ) = r_x*r_y * ( 1.0 - cos( omega ) );
            rot( 1 , 1 ) = cos( omega ) + r_y*r_y * ( 1.0 - cos( omega ) );
            rot( 1 , 2 ) = - r_x * sin( omega );
            
            rot( 2 , 0 ) = -r_y * sin( omega );
            rot( 2 , 1 ) = r_x * sin( omega );
            rot( 2 , 2 ) = cos( omega );
            
            // perform rotation
            s = prod( rot , s );

            s_x[n+1] = s(0);
            s_y[n+1] = s(1);
            s_z[n+1] = s(2);

            //std::clog << " = " << ( s_x[n+1]*b_x + s_y[n+1]*b_y + s_z[n+1]*b_z );

            //std::clog << " (" << s_x[n+1]*s_x[n+1] + s_y[n+1]*s_y[n+1] + s_z[n+1]*s_z[n+1] << ")" << std::endl;
            /*
            // what exactly should be lambda for the exp distr
            boost::exponential_distribution< value_type > exp_dist( b*beta );
            boost::variate_generator< boost::mt19937 & ,
                                      boost::exponential_distribution< value_type > > var_exp( m_rng , exp_dist );
            value_type cos_theta = var_exp( );

            
            // how to get a vector not perpendicular to b ?
            s_x[n+1] = cos_theta*b_x;
            s_y[n+1] = cos_theta*b_y;
            s_z[n+1] = cos_theta*b_z;
            */
        }
    }

private:
    const value_type m_q;
    const value_type m_nu;
};

#endif
