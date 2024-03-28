// Reference: [1] Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj. Programming Massively Parallel Processors: A Hands-on Approach, Forth Edition.

#include <iostream>
#include <numeric>
#include <memory>
#include <chrono>
#include "include/Matrix.cuh"

struct Stride
{
    __host__ __device__ Stride(std::size_t x_, std::size_t y_ = 0, std::size_t z_ = 0) : x{x_}, y{y_}, z{z_} {}
    std::size_t x, y, z;
};

template<class Type>
__global__ void fill(UniformMemory::Matrix<Type>& matrix, Type value)
{
    const std::size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const Stride stride{blockDim.x * gridDim.x, blockDim.y * gridDim.y};
    for(std::size_t i = x; i < matrix.row(); i += stride.x)
    {
        for(std::size_t j = y; j < matrix.column(); j += stride.y)
        {
            matrix(i, j) = value;
        }
    }
}

// tileWidth is assumed to be equal to blockDims 
constexpr std::size_t tileWidth = 32;

template<class Type>
__global__ void multiply(UniformMemory::Matrix<Type>& answer, const UniformMemory::Matrix<Type>& lhs, const UniformMemory::Matrix<Type>& rhs)
{
    __shared__ Type leftTile[tileWidth][tileWidth];
    __shared__ Type rightTile[tileWidth][tileWidth];
    
    const std::size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const Stride stride{blockDim.x * gridDim.x, blockDim.y * gridDim.y};

    const std::size_t commonSize = lhs.column();
    for(std::size_t i = x; i < lhs.row(); i += stride.x)
    {
        for(std::size_t j = y; j < rhs.column(); j += stride.y)
        {
            auto value = static_cast<Type>(0);
            for(std::size_t h = 0; h < commonSize / tileWidth; ++h)
            {
                if(threadIdx.y + h * tileWidth < commonSize) { leftTile[threadIdx.x][threadIdx.y] = lhs(i, threadIdx.y + h * tileWidth); }
                else { leftTile[threadIdx.x][threadIdx.y] = static_cast<Type>(0); }
                if(threadIdx.x + h * tileWidth < commonSize) { rightTile[threadIdx.x][threadIdx.y] = rhs(threadIdx.x + h * tileWidth, j); }
                else { rightTile[threadIdx.x][threadIdx.y] = static_cast<Type>(0); }
                __syncthreads();

                for(std::size_t k = 0; k < tileWidth; ++k)
                {
                    value += leftTile[threadIdx.x][k] * rightTile[k][threadIdx.y];
                }
                __syncthreads();
            }
            answer(i, j) = value;
        }
    }
}

template<class Type>
bool checkResult(const UniformMemory::Matrix<Type>& matrix, const Type value)
{
    for(std::size_t i = 0; i < matrix.row(); ++i)
    {
        for(std::size_t j = 0; j < matrix.column(); ++j)
        {
            if(matrix(i, j) != value) { return false; }
        }
    }
    return true;
}

int main()
{
    using MatrixType = UniformMemory::Matrix<double>;
    constexpr std::size_t lhsRow = 1024;
    constexpr std::size_t rhsCol = 1024;
    constexpr std::size_t common = 1024;
    auto pLHS = std::make_unique<MatrixType>(lhsRow, common);
    auto pRHS = std::make_unique<MatrixType>(common, rhsCol);
    {   // fill with ones
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid(32, 32);
        fill<<<blocksPerGrid, threadsPerBlock>>>(*pLHS, 1.0);
        fill<<<blocksPerGrid, threadsPerBlock>>>(*pRHS, 1.0);
    }
    auto pAnswer = std::make_unique<MatrixType>(lhsRow, rhsCol);
    {   // matrix multiplication takes 95ms according to nvprof achieving at least 10% improvement
        dim3 threadsPerBlock(tileWidth, tileWidth);
        dim3 blocksPerGrid(32, 32);
        multiply<<<blocksPerGrid ,threadsPerBlock>>>(*pAnswer, *pLHS, *pRHS);
        cudaDeviceSynchronize();
    }
    std::cout << std::boolalpha << checkResult(*pAnswer, static_cast<double>(common)) << "\n";
    return 0;
}
