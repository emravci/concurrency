// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

#pragma once

#include "Managed.cuh"

namespace ManagedMemory { template<class Type> class Matrix; }

template<class Type>
class ManagedMemory::Matrix : public Managed
{
    public:
    Matrix(std::size_t row, std::size_t column) : row_{row}, column_{column} { allocateUnifiedMemory(); }
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
        cudaMallocManaged(&data_, size() * sizeof(Type));
        cudaDeviceSynchronize();
    }
    void freeUnifiedMemory() 
    { 
        cudaDeviceSynchronize();
        cudaFree(data_); 
    }
    __host__ __device__ std::size_t convertToOneDimensionalIndex(std::size_t i, std::size_t j) const { return i * column_ + j; }
    private:
    Type *data_ = nullptr;
    std::size_t row_, column_;
};
