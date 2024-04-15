// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog
//            [2] CUDA C++ Programming Guide

// [-] make reference counting system-wide-thread-safe
// [-] without cudaDeviceSync at the beginning of dtor, results in seg-fault. why?
// [-] combine memory allocs into one single using void* then casting and shifting appropriately?..
// would it break alignment for alternating Type classes
// [x] ctor, copy ctor assignment op and dtor are not not needed on device (so far)

#pragma once

namespace UnifiedMemory { template<class Type> class SharedMatrix; }

template<class Type>
class UnifiedMemory::SharedMatrix
{
    public:
    SharedMatrix(std::size_t row, std::size_t column)
    {
        allocateUnifiedMemory(row * column);
        cudaDeviceSynchronize();
        initializeIntegrals(row, column);
    }
    SharedMatrix(const SharedMatrix& sharedMatrix) : data_{sharedMatrix.data_}, integrals_{sharedMatrix.integrals_}
    {
        ++mutableReferenceCount();
    }
    SharedMatrix& operator=(const SharedMatrix& sharedMatrix)
    {
        release();
        makeShallowCopy(sharedMatrix);
        ++mutableReferenceCount();
        return *this;
    }
    SharedMatrix(SharedMatrix&&) = delete;
    SharedMatrix& operator=(SharedMatrix&&) = delete;
    ~SharedMatrix() 
    { 
        cudaDeviceSynchronize();
        release(); 
    }
    __host__ __device__ std::size_t size() const { return row() * column(); }
    __host__ __device__ std::size_t row() const { return mutableRow(); }
    __host__ __device__ std::size_t column() const { return mutableColumn(); }
    __host__ __device__ Type& operator[](std::size_t i) { return data_[i]; }
    __host__ __device__ const Type& operator[](std::size_t i) const { return data_[i]; }
    __host__ __device__ Type* begin() { return data_; }
    __host__ __device__ Type* end() { return &data_[size()]; }
    __host__ __device__ const Type* cbegin() const { return data_; }
    __host__ __device__ const Type* cend() const { return &data_[size()]; }
    private:
    __host__ __device__ std::size_t& mutableRow() const { return integrals_[0]; }
    __host__ __device__ std::size_t& mutableColumn() const { return integrals_[1]; }
    __host__ __device__ std::size_t& mutableReferenceCount() const { return integrals_[2]; }
    void allocateUnifiedMemory(std::size_t size) 
    {
        cudaMallocManaged(&data_, size * sizeof(Type));
        cudaMallocManaged(&integrals_, 3 * sizeof(std::size_t));
    }
    void initializeIntegrals(std::size_t row, std::size_t column)
    {
        mutableRow() = row;
        mutableColumn() = column;
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
    void makeShallowCopy(const SharedMatrix& sharedMatrix)
    {
        data_ = sharedMatrix.data_;
        integrals_ = sharedMatrix.integrals_;
    }
    private:
    Type *data_ = nullptr;
    std::size_t *integrals_ = nullptr;
};
