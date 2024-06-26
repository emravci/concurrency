// References: [1] CUDA C++ Programming Guide
//             [2] CUDA C++ Best Practices Guide
//             [3] CUDA Training Series at Oak Ridge National Laboratory

// Assumption: [1] Instances of the class are assumed to be accessed by device#0. 
//                 This improves page-faults, when automatic variable instance is passed-by-ref.

#pragma once

#include "error.cuh"

namespace Deus { template<class Type> class Matrix; }

template<class Type>
class Deus::Matrix
{
    public:
    using SelfType = Matrix<Type>;
    using ValueType = Type;
    using SizeType = std::size_t;
    Matrix(SizeType row, SizeType column) : row_{row}, column_{column}
    { 
        allocateUnifiedMemory();
        adviseOnMemory();
    }
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) = delete;
    Matrix& operator=(Matrix&&) = delete;
    ~Matrix() 
    { 
        freeUnifiedMemory(); 
        adviseAgainstMemory();
    }
    void prefetchAllAsync(int to, cudaStream_t stream = 0) const
    {
        checkCudaError(cudaMemPrefetchAsync(data_, size() * sizeof(ValueType), to, stream));
        checkCudaError(cudaMemPrefetchAsync(&row_, sizeof(SizeType), to, stream));
        checkCudaError(cudaMemPrefetchAsync(&column_, sizeof(SizeType), to, stream));
    }
    void prefetchWholeDataAsync(int to, cudaStream_t stream = 0) const
    {
        checkCudaError(cudaMemPrefetchAsync(data_, size() * sizeof(ValueType), to, stream));
    }
    __host__ __device__ SizeType size() const { return row_ * column_; }
    __host__ __device__ const SizeType& row() const { return row_; }
    __host__ __device__ const SizeType& column() const { return column_; }
    __host__ __device__ ValueType& operator()(SizeType i, SizeType j) { return data_[convertToOneDimensionalIndex(i, j)]; }
    __host__ __device__ const ValueType& operator()(SizeType i, SizeType j) const { return data_[convertToOneDimensionalIndex(i, j)]; }
    __host__ __device__ ValueType* begin() { return data_; }
    __host__ __device__ ValueType* end() { return &data_[size()]; }
    __host__ __device__ const ValueType* cbegin() const { return data_; }
    __host__ __device__ const ValueType* cend() const { return &data_[size()]; }
    private:
    void allocateUnifiedMemory() { checkCudaError(cudaMallocManaged(&data_, size() * sizeof(ValueType))); }
    void freeUnifiedMemory() 
    { 
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(data_));
    }
    __host__ __device__ SizeType convertToOneDimensionalIndex(SizeType i, SizeType j) const { return i * column_ + j; }
    void adviseOnMemory() const { checkCudaError(cudaMemAdvise(this, sizeof(SelfType), cudaMemAdviseSetAccessedBy, 0)); }
    void adviseAgainstMemory() const { checkCudaError(cudaMemAdvise(this, sizeof(SelfType), cudaMemAdviseUnsetAccessedBy, 0)); }
    private:
    ValueType *data_ = nullptr;
    SizeType row_, column_;
};
