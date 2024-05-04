#include <assert.h>

namespace Deus 
{
    template<class ValueType, class SizeType, class OperatorType> struct Reduce;
    __host__ __device__ bool isExactPowerOfTwo(std::size_t);
    template<class Type> struct Add;
}

__host__ __device__ bool Deus::isExactPowerOfTwo(std::size_t size) { return (size & (size - 1)) == 0; }

template<class Type>
struct Deus::Add
{
    __device__ void operator()(Type &lhs, const Type &rhs) { lhs += rhs; }
};

template<class ValueType, class SizeType, class OperatorType = Deus::Add<ValueType>>
struct Deus::Reduce
{   // for reduction size must be exact power of 2
    __device__ Reduce(ValueType *array, SizeType size, OperatorType fcn = {}) : array_{array}, size_{size}, operator_{fcn} 
    {
        assert(isExactPowerOfTwo(size_));
    }
    __device__ void operator()(SizeType index)
    {
        for(SizeType half = size_ / 2; half != 0; half /= 2)
        {
            if(index < half) { operator_(array_[index], array_[index + half]); }
            __syncthreads();
        }
    }
    ValueType *array_;
    SizeType size_;
    OperatorType operator_;
};
