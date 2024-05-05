// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

// Assumption: [1] Instances of the class are assumed to be accessed by device#0. 
//                 This improves page-faults, when automatic variable instance is passed-by-ref.

#pragma once

#include "Managed.cuh"
#include "../Deus/error.cuh"

namespace ManagedMemory { template<class Type> class Matrix; }

template<class Type>
class ManagedMemory::Matrix : public Managed
{
    public:
    using SelfType = Matrix<Type>;
    Matrix(std::size_t row, std::size_t column) : row_{row}, column_{column} 
    { 
        allocateUnifiedMemory();
        advise();
    }
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) = delete;
    Matrix& operator=(Matrix&&) = delete;
    ~Matrix() { freeUnifiedMemory(); }
    __host__ __device__ std::size_t size() const { return row_ * column_; }
    __host__ __device__ std::size_t row() const { return row_; }
    __host__ __device__ std::size_t column() const { return column_; }
    __host__ __device__ Type& operator()(std::size_t i, std::size_t j) { return data_[convertToOneDimensionalIndex(i, j)]; }
    __host__ __device__ const Type& operator()(std::size_t i, std::size_t j) const { return data_[convertToOneDimensionalIndex(i, j)]; }
    __host__ __device__ Type* begin() { return data_; }
    __host__ __device__ Type* end() { return &data_[size()]; }
    __host__ __device__ const Type* cbegin() const { return data_; }
    __host__ __device__ const Type* cend() const { return &data_[size()]; }
    private:
    void allocateUnifiedMemory() 
    {   
        checkCudaError(cudaMallocManaged(&data_, size() * sizeof(Type)));
        checkCudaError(cudaDeviceSynchronize());
    }
    void freeUnifiedMemory() 
    { 
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(data_));
    }
    __host__ __device__ std::size_t convertToOneDimensionalIndex(std::size_t i, std::size_t j) const { return i * column_ + j; }
    void advise() const { checkCudaError(cudaMemAdvise(this, sizeof(SelfType), cudaMemAdviseSetAccessedBy, 0)); }
    private:
    Type *data_ = nullptr;
    std::size_t row_, column_;
};
