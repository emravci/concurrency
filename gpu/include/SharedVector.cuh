// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog
//            [2] CUDA C++ Programming Guide

// [-] make reference counting thread-safe
// [-] without cudaDeviceSync at the beginning of dtor, results in seg-fault. why?
// [-] combine memory allocs into one single using void* then casting and shifting appropriately?..
// would it break alignment for alternating Type classes
// [x] ctor, copy ctor assignment op and dtor are not not needed on device (so far)

#pragma once

namespace UnifiedMemory { template<class Type> class SharedVector; }

template<class Type>
class UnifiedMemory::SharedVector
{
    public:
    SharedVector(std::size_t size)
    {
        allocateUnifiedMemory(size);
        cudaDeviceSynchronize();
        initializeIntegrals(size);
    }
    SharedVector(const SharedVector& sharedVector) : data_{sharedVector.data_}, integrals_{sharedVector.integrals_}
    {
        ++mutableReferenceCount();
    }
    SharedVector& operator=(const SharedVector& sharedVector)
    {
        release();
        makeShallowCopy(sharedVector);
        ++mutableReferenceCount();
        return *this;
    }
    SharedVector(SharedVector&&) = delete;
    SharedVector& operator=(SharedVector&&) = delete;
    ~SharedVector() 
    { 
        cudaDeviceSynchronize();
        release(); 
    }
    __host__ __device__ std::size_t size() const { return mutableSize(); }
    __host__ __device__ Type& operator[](std::size_t i) { return data_[i]; }
    __host__ __device__ const Type& operator[](std::size_t i) const { return data_[i]; }
    __host__ __device__ Type* begin() { return data_; }
    __host__ __device__ Type* end() { return &data_[size()]; }
    __host__ __device__ const Type* cbegin() const { return data_; }
    __host__ __device__ const Type* cend() const { return &data_[size()]; }
    private:
    __host__ __device__ std::size_t& mutableSize() const { return integrals_[0]; }
    __host__ __device__ std::size_t& mutableReferenceCount() const { return integrals_[1]; }
    void allocateUnifiedMemory(std::size_t size) 
    {
        cudaMallocManaged(&data_, size * sizeof(Type));
        cudaMallocManaged(&integrals_, 2 * sizeof(std::size_t));
    }
    void initializeIntegrals(std::size_t size)
    {
        mutableSize() = size;
        mutableReferenceCount() = 1;
    }
    void release()
    {
        --mutableReferenceCount();
        if(mutableReferenceCount() == 0) { freeUnifiedMemory(); }
    }
    void freeUnifiedMemory() 
    {
        cudaFree(data_);
        cudaFree(integrals_);
    }
    void makeShallowCopy(const SharedVector& sharedVector)
    {
        data_ = sharedVector.data_;
        integrals_ = sharedVector.integrals_;
    }
    private:
    Type *data_ = nullptr;
    std::size_t *integrals_ = nullptr;
};
