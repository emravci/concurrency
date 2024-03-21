// Reference: [1] An Even Easier Introduction to CUDA by Mark Harris
//            [2] Unified Memory for CUDA Beginners by Mark Harris
//            [3] Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming.


#include <iostream>
#include <numeric>
#include <chrono>

const int N = 1024 * 1024 * 256;
const int threadsPerBlock = 1024;
const int blocksPerGrid = 256;

template<class Type>
class Vector
{
    public:
    Vector(std::size_t size) : data_{nullptr}, size_{size} { allocateUnifiedMemory(); }
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&&) = default;
    Vector& operator=(Vector&&) = default;
    ~Vector() { freeUnifiedMemory(); }
    std::size_t size() const { return size_; }
    Type* begin() { return &data_[0]; }
    Type* end() { return &data_[size_]; }
    Type* address() { return data_; }
    Type& operator[](std::size_t i) { return data_[i]; }
    const Type& operator[](std::size_t i) const { return data_[i]; }
    private:
    void allocateUnifiedMemory() { cudaMallocManaged(&data_, size_ * sizeof(Type)); }
    void freeUnifiedMemory() { cudaFree(data_); }
    private:
    Type *data_;
    std::size_t size_;
};


template<class Type>
__global__ void init(std::size_t N, Type *array)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) { array[i] = static_cast<Type>(1); }
}

template<class Type>
__global__ void dot(std::size_t N, Type *a, Type *b, Type *c) 
{
    __shared__ Type cache[threadsPerBlock];,

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;

    auto partial = static_cast<Type>(0);
    for(int i = index; i < N; i += stride) { partial += a[index] * b[index]; }
    cache[cacheIndex] = partial;
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    for(int half = blockDim.x / 2; half != 0; half /= 2)
    {
        if (cacheIndex < half) { cache[cacheIndex] += cache[cacheIndex + half]; }
        __syncthreads();
    }

    if (cacheIndex == 0) { c[blockIdx.x] = cache[0]; }
}

int main()
{
    Vector<double> vector(N);
    // std::iota(vector.begin(), vector.end(), 0.0);
    init<<<256, 1024>>>(vector.size(), vector.address());
    cudaDeviceSynchronize();

	// concurrent - 9ms but memory allocation takes around 2sec
    Vector<double> partial(blocksPerGrid);
    auto begin = std::chrono::high_resolution_clock::now();
    dot<<<blocksPerGrid, threadsPerBlock>>>(vector.size(), vector.address(), vector.address(), partial.address());
    cudaDeviceSynchronize();
    double value = std::accumulate(partial.begin(), partial.end(), 0.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << std::fixed << value << " in " << duration.count() << "ms\n";

    return 0;
}
