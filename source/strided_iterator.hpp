#ifndef THRUST_STRIDED_RANGE
#define THRUST_STRIDED_RANGE

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

namespace thrust {
template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

    /*
template< typename DifferenceType >
struct stride_functor : public thrust::unary_function< DifferenceType , DifferenceType >
{
    typedef DifferenceType difference_type;

    difference_type stride;
    
    stride_functor(difference_type stride)
        : stride(stride) {}
    
    __host__ __device__
    difference_type operator()(const difference_type& i) const
    { 
        return stride * i;
    }
};

template< class Iterator >
    //calculate return type....
thrust::permutation_iterator< Iterator , 
                              thrust::transform_iterator< stride_functor<typename thrust::iterator_difference<Iterator>::type> , 
                                                          thrust::counting_iterator<difference_type> > >
make_strided_iterator( Iterator iter , const int stride )
{

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;
    typedef typename thrust::counting_iterator<difference_type> CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor<difference_type> , CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator> PermutationIterator;


    return PermutationIterator(iter, TransformIterator(CountingIterator(0), stride_functor<difference_type>(stride)));
    } */

} //namespace thrust
#endif
