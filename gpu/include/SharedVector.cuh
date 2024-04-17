// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog
//            [2] CUDA C++ Programming Guide

// [-] make reference counting thread-safe
// [-] without cudaDeviceSync at the beginning of dtor, results in seg-fault. why?
// [x] combined memory allocs into one single using void* then casting and shifting appropriately?..
// would it break alignment for alternating Type classes
// [x] ctor, copy ctor assignment op and dtor are not not needed on device (so far)

#pragma once

namespace UnifiedMemory { template<class Type> class SharedVector; }

template<class Type>
class UnifiedMemory::SharedVector
{
    public:
    SharedVector(std::size_t size) : size_{size}
    {
        allocateUnifiedMemory();
        cudaDeviceSynchronize();
        referenceCount() = 1;
    }
    SharedVector(const SharedVector& sharedVector) : data_{sharedVector.data_}, size_{sharedVector.size_}
    {
        ++referenceCount();
    }
    SharedVector& operator=(const SharedVector& sharedVector)
    {
        release();
        makeShallowCopy(sharedVector);
        ++referenceCount();
        return *this;
    }
    SharedVector(SharedVector&&) = delete;
    SharedVector& operator=(SharedVector&&) = delete;
    ~SharedVector() 
    { 
        cudaDeviceSynchronize();
        release(); 
    }
    __host__ __device__ std::size_t size() const { return size_; }
    __host__ __device__ Type& operator[](std::size_t i) { return elements()[i]; }
    __host__ __device__ const Type& operator[](std::size_t i) const { return elements()[i]; }
    __host__ __device__ Type* begin() { return elements(); }
    __host__ __device__ Type* end() { return &elements()[size()]; }
    __host__ __device__ const Type* cbegin() const { return elements(); }
    __host__ __device__ const Type* cend() const { return &elements()[size()]; }
    private:
    __host__ __device__ std::size_t& referenceCount() const { return static_cast<std::size_t*>(data_)[0]; }
    __host__ __device__ Type* elements() const { return reinterpret_cast<Type*>(&static_cast<std::size_t*>(data_)[1]); }
    void allocateUnifiedMemory() { cudaMallocManaged(&data_, sizeof(std::size_t) + size_ * sizeof(Type)); }
    void release() { if(--referenceCount() == 0) { freeUnifiedMemory(); } }
    void freeUnifiedMemory() { cudaFree(data_); }
    void makeShallowCopy(const SharedVector& sharedVector)
    {
        data_ = sharedVector.data_;
        size_ = sharedVector.size_;
    }
    private:
    void *data_ = nullptr;
    std::size_t size_ = 0;
};
