// Reference: [1] CUDA C++ Programming Guide

#pragma once

namespace ManagedMemory 
{ 
    template<class Type> class SharedMatrix;
    template<class Type> class TransposeView;
    template<class Type> TransposeView<Type> transpose(const SharedMatrix<Type>&);
}

template<class Type>
class ManagedMemory::TransposeView
{
    public:
    TransposeView(const SharedMatrix<Type>& sharedMatrix) : sharedMatrix_{sharedMatrix} {}
    __host__ __device__ std::size_t size() const { return sharedMatrix_.size(); }
    __host__ __device__ std::size_t row() const { return sharedMatrix_.column(); }
    __host__ __device__ std::size_t column() const { return sharedMatrix_.row(); }
    __host__ __device__ Type& operator()(std::size_t i, std::size_t j) { return sharedMatrix_[convertToOneDimensionalIndex(i, j)]; }
    __host__ __device__ const Type& operator()(std::size_t i, std::size_t j) const { return sharedMatrix_[convertToOneDimensionalIndex(i, j)]; }
    private:
    __host__ __device__ std::size_t convertToOneDimensionalIndex(std::size_t i, std::size_t j) const { return i + row() * j; }
    SharedMatrix<Type> sharedMatrix_;
};

template<class Type>
ManagedMemory::TransposeView<Type> ManagedMemory::transpose(const SharedMatrix<Type>& sharedMatrix)
{
    return TransposeView<Type>{sharedMatrix};
}
