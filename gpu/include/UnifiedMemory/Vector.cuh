// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

#pragma once

#include "Managed.cuh"

namespace UnifiedMemory { template<class Type> class Vector; }

template<class Type>
class UnifiedMemory::Vector : public Managed
{
    public:
    using ValueType = Type;
    using SizeType = std::size_t;
    Vector(SizeType size) : size_{size} { allocateUnifiedMemory(); }
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
        cudaMallocManaged(&data_, size_ * sizeof(ValueType));
        cudaDeviceSynchronize();
    }
    void freeUnifiedMemory() 
    {
        cudaDeviceSynchronize();
        cudaFree(data_); 
    }
    private:
    ValueType *data_ = nullptr;
    SizeType size_ = 0;
};
