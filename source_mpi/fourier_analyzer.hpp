#ifndef FOURIER_ANALYZER
#define FOURIER_ANALYZER

#include <iostream>
#include <vector>
#include <math.h>

#include <thrust/inner_product.h>

template< class VectorType , class ValueType >
class fourier_analyzer
{
public:
    
    typedef VectorType vector_type;
    typedef ValueType value_type;

    fourier_analyzer( const int N , const value_type q )
        : m_co( N ) , m_si( N )
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
    }

    value_type analyze( const vector_type &energy )
    {
       
        const value_type real = thrust::inner_product( energy.begin() , 
                                                       energy.end() , 
                                                       m_co.begin() , 
                                                       0.0 );
        
        const value_type imag = thrust::inner_product( energy.begin() , 
                                                       energy.end() , 
                                                       m_si.begin() , 
                                                       0.0 );
        return real*real + imag*imag;
    }

private:
    vector_type m_co;
    vector_type m_si;
};

#endif
