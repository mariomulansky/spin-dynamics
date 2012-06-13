#ifndef FOURIER_ANALYZER
#define FOURIER_ANALYZER

#include <iostream>
#include <vector>
#include <cmath>

#include <thrust/inner_product.h>

template< class VectorType , class ValueType >
class fourier_analyzer
{
public:
    
    typedef VectorType vector_type;
    typedef ValueType value_type;

    fourier_analyzer( const value_type q , const int N )
        : m_q( q ) , m_co( N ) , m_si( N )
    { 
        std::vector< value_type > co( N );
        std::vector< value_type > si( N );
        for( int n=0 ; n<N ; ++n )
        {
            using std::cos;
            using std::sin;

            co[n] = cos( q*n / N );
            si[n] = sin( q*n / N );
        }
        thrust::copy( co.begin() , co.end() , m_co.begin() );
        thrust::copy( si.begin() , si.end() , m_si.begin() );
    }

    value_type analyze( const vector_type &s_x , 
                        const vector_type &s_y , 
                        const vector_type &s_z , 
                        const vector_type &b_x , 
                        const vector_type &b_y , 
                        const vector_type &b_z , 
                        vector_type &b_norm )
    {
        //calculate energies and store result in b_norm
        thrust::for_each(
            make_zip_iterator( make_tuple(
                s_x.begin()+1 ,
                s_y.begin()+1 ,
                s_y.begin()+1 ,
                b_x.begin() ,
                b_y.begin() ,
                b_z.begin() ,
                b_norm.begin() ) ),
            make_zip_iterator( make_tuple(
                s_x.end()-1 ,
                s_y.end()-1 ,
                s_y.end()-1 ,
                b_x.end() ,
                b_y.end() ,
                b_z.end() ,
                b_norm.end() ) ) ,
            energy_functor< value_type >() );

        std::clog << s_x[5]*s_x[5] + s_y[5]*s_y[5] + s_z[5]*s_z[5] << " , ";
        std::clog << thrust::reduce( b_norm.begin() , b_norm.end() ) << std::endl;

        const value_type real = thrust::inner_product( b_norm.begin() , 
                                                       b_norm.end() , 
                                                       m_co.begin() , 
                                                       0.0 );
        
        const value_type imag = thrust::inner_product( b_norm.begin() , 
                                                       b_norm.end() , 
                                                       m_si.begin() , 
                                                       0.0 );
        
        return real*real + imag*imag;
    }

private:
    value_type m_q;
    vector_type m_co;
    vector_type m_si;
};

#endif
