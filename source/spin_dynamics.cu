#include <iostream>
#include <boost/random.hpp>

#include <thrust/device_vector.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "spin_stepper.hpp"
#include "fourier_analyzer.hpp"

typedef double value_type;
typedef thrust::device_vector< value_type > device_type;

typedef std::vector< value_type > host_type;

static const int N = 10;
static const int steps = 100;
static const double dt = 0.1;
static const double q = 25.0;
static const double beta = 0.1;

int main()
{
    boost::mt19937 m_rng;
    boost::uniform_01<> m_dist;

    /* define initial conditions */
    host_type s_x_host( N+2 );
    host_type s_y_host( N+2 );
    host_type s_z_host( N+2 );

    host_type h_x_host( N );
    host_type h_y_host( N );
    host_type h_z_host( N );

    s_x_host[0] = 0.0; s_x_host[N+1] = 0.0; 
    s_y_host[0] = 0.0; s_y_host[N+1] = 0.0;
    s_z_host[0] = 0.0; s_z_host[N+1] = 0.0;
    /* generate random initial conditions */
    for( int n=0 ; n<N ; ++n )
    {
        s_x_host[n+1] = m_dist( m_rng ) + beta * cos( q*n / N );
        s_y_host[n+1] = m_dist( m_rng ) + beta * cos( q*n / N );
        s_z_host[n+1] = m_dist( m_rng ) + beta * cos( q*n / N );
        value_type nrm = sqrt( s_x_host[n+1]*s_x_host[n+1] + 
                               s_y_host[n+1]*s_y_host[n+1] + 
                               s_z_host[n+1]*s_z_host[n+1] );
        s_x_host[n+1] /= nrm;
        s_y_host[n+1] /= nrm;
        s_z_host[n+1] /= nrm;
        
        h_x_host[n] = m_dist( m_rng );
        h_y_host[n] = m_dist( m_rng );
        h_z_host[n] = m_dist( m_rng );

        nrm = sqrt( h_x_host[n]*h_x_host[n] + 
                    h_y_host[n]*h_y_host[n] + 
                    h_z_host[n]*h_z_host[n] );
        h_x_host[n] /= nrm;
        h_y_host[n] /= nrm;
        h_z_host[n] /= nrm;

    }
    std::clog << "initialization finished" << std::endl;

    // initialize device vectors
    std::clog << "copy to device..." << std::endl;

    // vectors s_*_host have length N+2 to account for boundary conditions
    device_type s_x( s_x_host.begin() , s_x_host.end() );
    device_type s_y( s_y_host.begin() , s_y_host.end() );
    device_type s_z( s_z_host.begin() , s_z_host.end() );
    
    device_type b_x( N );
    device_type b_y( N );
    device_type b_z( N );
    device_type b_norm( N );
    
    device_type h_x( h_x_host.begin() , h_x_host.end() );
    device_type h_y( h_y_host.begin() , h_y_host.end() );
    device_type h_z( h_z_host.begin() , h_z_host.end() );
    
    spin_stepper< device_type , value_type > stepper( N , h_x , h_y , h_z );
    fourier_analyzer< device_type , value_type > fourier( N , q );

    for( int n=0 ; n<steps ; ++n )
    {
        stepper.compute_b( s_x , s_y , s_z , 
                         b_x , b_y , b_z , 
                         b_norm );
        if( (n%10) == 0 )
        {
            value_type f_q = fourier.analyze( s_x , s_y , s_z , b_x , b_y , b_z , b_norm );
            std::cout << n*dt << '\t' << f_q << std::endl;
        }
        stepper.do_step( s_x , s_y , s_z , 
                         b_x , b_y , b_z , 
                         b_norm , dt );
    }

}
