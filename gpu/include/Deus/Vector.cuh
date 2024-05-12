// References: [1] CUDA C++ Programming Guide
//             [2] CUDA C++ Best Practices Guide
//             [3] CUDA Training Series at Oak Ridge National Laboratory

// Assumption: [1] Instances of the class are assumed to be accessed by device#0. 
//                 This improves page-faults, when automatic variable instance is passed-by-ref.

#pragma once

#include "error.cuh"

namespace Deus { template<class Type> class Vector; }

template<class Type>
class Deus::Vector
{
    public:
    using SelfType = Vector<Type>;
    using ValueType = Type;
    using SizeType = std::size_t;
    Vector(SizeType size) : size_{size} 
    {
        allocateUnifiedMemory();
        adviseOnMemory();
    }
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&&) = delete;
    Vector& operator=(Vector&&) = delete;
    ~Vector() 
    {
        adviseAgainstMemory();
        freeUnifiedMemory(); 
    }
    void prefetchAllAsync(int to, cudaStream_t stream = 0) const
    {
        checkCudaError(cudaMemPrefetchAsync(data_, size_ * sizeof(ValueType), to, stream));
        checkCudaError(cudaMemPrefetchAsync(&size_, sizeof(SizeType), to, stream));
    }
    void prefetchWholeDataAsync(int to, cudaStream_t stream = 0) const
    {
        checkCudaError(cudaMemPrefetchAsync(data_, size_ * sizeof(ValueType), to, stream));
    }
    __host__ __device__ const SizeType& size() const { return size_; }
    __host__ __device__ ValueType& operator[](SizeType i) { return data_[i]; }
    __host__ __device__ const ValueType& operator[](SizeType i) const { return data_[i]; }
    __host__ __device__ ValueType* begin() { return data_; }
    __host__ __device__ ValueType* end() { return &data_[size_]; }
    __host__ __device__ const ValueType* cbegin() const { return data_; }
    __host__ __device__ const ValueType* cend() const { return &data_[size_]; }
    private:
    void allocateUnifiedMemory() 
    {
        checkCudaError(cudaMallocManaged(&data_, size_ * sizeof(ValueType)));
        checkCudaError(cudaDeviceSynchronize());
    }
    void freeUnifiedMemory() 
    {
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(data_)); 
    }
    void adviseOnMemory() const { checkCudaError(cudaMemAdvise(this, sizeof(SelfType), cudaMemAdviseSetAccessedBy, 0)); }
    void adviseAgainstMemory() const { checkCudaError(cudaMemAdvise(this, sizeof(SelfType), cudaMemAdviseUnsetAccessedBy, 0)); }
    private:
    ValueType *data_ = nullptr;
    SizeType size_ = 0;
};
