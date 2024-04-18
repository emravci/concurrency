// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog
//            [2] CUDA C++ Programming Guide

// [-] make reference counting system-wide-thread-safe
// [-] without cudaDeviceSync at the beginning of dtor, results in seg-fault. why?
// [x] combine memory allocs into one single using void* then casting and shifting appropriately?..
// would it break alignment for alternating Type classes
// [x] ctor, copy ctor assignment op and dtor are not not needed on device (so far)

#pragma once

namespace UnifiedMemory { template<class Type> class SharedMatrix; }

template<class Type>
class UnifiedMemory::SharedMatrix
{
    public:
    SharedMatrix(std::size_t row, std::size_t column) : row_{row}, column_{column}
    {
        allocateUnifiedMemory();
        cudaDeviceSynchronize();
        referenceCount() = 1;
    }
    SharedMatrix(const SharedMatrix& sharedMatrix) : data_{sharedMatrix.data_}, row_{sharedMatrix.row_}, column_{sharedMatrix.column_}
    {
        ++referenceCount();
    }
    SharedMatrix& operator=(const SharedMatrix& sharedMatrix)
    {
        release();
        makeShallowCopy(sharedMatrix);
        ++referenceCount();
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
    __host__ __device__ std::size_t row() const { return row_; }
    __host__ __device__ std::size_t column() const { return column_; }
    __host__ __device__ Type& operator[](std::size_t i) { return elements()[i]; }
    __host__ __device__ const Type& operator[](std::size_t i) const { return elements()[i]; }
    __host__ __device__ Type& operator()(std::size_t i, std::size_t j) { return elements()[convertToOneDimensionalIndex(i, j)]; }
    __host__ __device__ const Type& operator()(std::size_t i, std::size_t j) const { return elements()[convertToOneDimensionalIndex(i, j)]; }
    __host__ __device__ Type* begin() { return elements(); }
    __host__ __device__ Type* end() { return &elements()[size()]; }
    __host__ __device__ const Type* cbegin() const { return elements(); }
    __host__ __device__ const Type* cend() const { return &elements()[size()]; }
    private:
    __host__ __device__ std::size_t& referenceCount() const { return static_cast<std::size_t*>(data_)[0]; }
    __host__ __device__ Type* elements() const { return reinterpret_cast<Type*>(&static_cast<std::size_t*>(data_)[1]); }
    __host__ __device__ std::size_t convertToOneDimensionalIndex(std::size_t i, std::size_t j) const { return i * column() + j; }
    void allocateUnifiedMemory() { cudaMallocManaged(&data_, sizeof(std::size_t) + size() * sizeof(Type)); }
    void release() { if(--referenceCount() == 0) { freeUnifiedMemory(); } }
    void freeUnifiedMemory() { cudaFree(data_); }    
    void makeShallowCopy(const SharedMatrix& sharedMatrix)
    {
        data_ = sharedMatrix.data_;
        row_ = sharedMatrix.row_;
        column_ = sharedMatrix.column_;
    }
    private:
    void *data_ = nullptr;
    std::size_t row_ = 0, column_ = 0;
};
