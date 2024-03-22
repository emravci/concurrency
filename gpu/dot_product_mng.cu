// Reference: [1] Unified Memory in CUDA 6 by Mark Harris
//            [2] An Even Easier Introduction to CUDA by Mark Harris
//            [3] Unified Memory for CUDA Beginners by Mark Harris
//            [4] Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming.


#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>

const int N = 1024 * 1024 * 256;
const int threadsPerBlock = 1024;
const int blocksPerGrid = 256;

class Managed
{
    public: 
    void *operator new(std::size_t len)
    {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }
    void operator delete(void *ptr)
    {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

template<class Type>
class ManagedVector : public Managed
{
    public:
    ManagedVector(std::size_t size) : data_{nullptr}, size_{size} { allocateUnifiedMemory(); }
    ManagedVector(const ManagedVector&) = delete;
    ManagedVector& operator=(const ManagedVector&) = delete;
    ManagedVector(ManagedVector&&) = delete;
    ManagedVector& operator=(ManagedVector&&) = delete;
    ~ManagedVector() { freeUnifiedMemory(); }
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
    Type *data_;
    std::size_t size_;
};

template<class Type>
__global__ void init(ManagedVector<Type>& array)
{
    const std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    const std::size_t N = array.size();
    for (std::size_t i = index; i < N; i += stride) { array[i] = static_cast<Type>(1); }
}

template<class Type>
__global__ void dot(ManagedVector<Type>& partials, const ManagedVector<Type>& lhs, const ManagedVector<Type>& rhs)
{
    __shared__ Type cache[threadsPerBlock];

    const std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    const std::size_t cacheIndex = threadIdx.x;

    auto partial = static_cast<Type>(0);
    for(std::size_t i = index, N = lhs.size(); i < N; i += stride) { partial += lhs[index] * rhs[index]; }
    cache[cacheIndex] = partial;
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    for(std::size_t half = blockDim.x / 2; half != 0; half /= 2)
    {
        if (cacheIndex < half) { cache[cacheIndex] += cache[cacheIndex + half]; }
        __syncthreads();
    }

    if(cacheIndex == 0) { partials[blockIdx.x] = cache[0]; }
}

int main()
{
    auto ptrVectorOfOnes = std::make_unique<ManagedVector<double>>(N);
    init<<<256, 1024>>>(*ptrVectorOfOnes);
    cudaDeviceSynchronize();

    // 9ms total
    // concurrent - 3ms dot according to nvprof but memory allocation takes around 2sec
    auto ptrPartials = std::make_unique<ManagedVector<double>>(blocksPerGrid);
    auto begin = std::chrono::high_resolution_clock::now();
    dot<<<blocksPerGrid, threadsPerBlock>>>(*ptrPartials, *ptrVectorOfOnes, *ptrVectorOfOnes);
    cudaDeviceSynchronize();
    // sequential addition on CPU - 6ms probably
    double value = std::accumulate(ptrPartials->cbegin(), ptrPartials->cend(), 0.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << std::fixed << value << " in " << duration.count() << "ms\n";

    return 0;
}
