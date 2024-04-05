// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

#pragma once

#include "Managed.cuh"

namespace UniformMemory { template<class Type> class Vector; }

template<class Type>
class UniformMemory::Vector : public Managed
{
    public:
    Vector(std::size_t size) : size_{size} { allocateUnifiedMemory(); }
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&&) = delete;
    Vector& operator=(Vector&&) = delete;
    ~Vector() { freeUnifiedMemory(); }
    __host__ __device__ std::size_t size() const { return size_; }
    __host__ __device__ Type& operator[](std::size_t i) { return data_[i]; }
    __host__ __device__ const Type& operator[](std::size_t i) const { return data_[i]; }
    __host__ __device__ Type* begin() { return data_; }
    __host__ __device__ Type* end() { return &data_[size_]; }
    __host__ __device__ const Type* cbegin() const { return data_; }
    __host__ __device__ const Type* cend() const { return &data_[size_]; }
    private:
    void allocateUnifiedMemory() { cudaMallocManaged(&data_, size_ * sizeof(Type)); }
    void freeUnifiedMemory() { cudaFree(data_); }
    private:
    Type *data_ = nullptr;
    std::size_t size_;
};
