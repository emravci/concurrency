// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

// Assumption: [1] Instances of the class are assumed to be accessed by device#0. 
//                 This improves page-faults, when automatic variable instance is passed-by-ref.

#pragma once

#include "Managed.cuh"
#include "../Deus/error.cuh"

namespace ManagedMemory { template<class Type> class Vector; }

template<class Type>
class ManagedMemory::Vector : public Managed
{
    public:
    using SelfType = Vector<Type>;
    using ValueType = Type;
    using SizeType = std::size_t;
    Vector(SizeType size) : size_{size} 
    { 
        allocateUnifiedMemory();
        advise();
    }
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&&) = delete;
    Vector& operator=(Vector&&) = delete;
    ~Vector() { freeUnifiedMemory(); }
    __host__ __device__ SizeType size() const { return size_; }
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
    void advise() const { checkCudaError(cudaMemAdvise(this, sizeof(SelfType), cudaMemAdviseSetAccessedBy, 0)); }
    private:
    ValueType *data_ = nullptr;
    SizeType size_ = 0;
};
