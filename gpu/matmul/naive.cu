// Reference: [1] Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj. Programming Massively Parallel Processors: A Hands-on Approach, Forth Edition.
//            [2] Unified Memory for CUDA Beginners by Mark Harris

#include <iostream>
#include <numeric>
#include <memory>
#include <chrono>
#include "include/Matrix.cuh"

template<class Type>
__global__ void fill(UniformMemory::Matrix<Type>& matrix, Type value)
{
    for(std::size_t i = threadIdx.y + blockIdx.y * blockDim.y; i < matrix.row(); i += blockDim.y * gridDim.y)
    {
        for(std::size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < matrix.column(); j += blockDim.x * gridDim.x)
        {
            matrix(i, j) = value;
        }
    }
}

template<class Type>
__global__ void multiply(UniformMemory::Matrix<Type>& answer, const UniformMemory::Matrix<Type>& lhs, const UniformMemory::Matrix<Type>& rhs)
{
    for(std::size_t i = threadIdx.y + blockIdx.y * blockDim.y; i < lhs.row(); i += blockDim.y * gridDim.y)
    {
        for(std::size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < rhs.column(); j += blockDim.x * gridDim.x)
        {
            auto value = static_cast<Type>(0);
            for(std::size_t k = 0; k < lhs.column() && k < rhs.row(); ++k)
            {
                value += lhs(i, k) * rhs(k, j);
            }
            answer(i, j) = value;
        }
    }
}

template<class Type>
void multiplyOnHost(UniformMemory::Matrix<Type>& answer, const UniformMemory::Matrix<Type>& lhs, const UniformMemory::Matrix<Type>& rhs)
{
    for(std::size_t i = 0; i < lhs.row(); ++i)
    {
        for(std::size_t j = 0; j < rhs.column(); ++j)
        {
            auto value = static_cast<Type>(0);
            for(std::size_t k = 0; k < lhs.column() && k < rhs.row(); ++k)
            {
                value += lhs(i, k) * rhs(k, j);
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
    {   // matrix multiplication takes 92ms according to nvprof
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid(32, 32);
        multiply<<<blocksPerGrid ,threadsPerBlock>>>(*pAnswer, *pLHS, *pRHS);
        cudaDeviceSynchronize();
    }
    std::cout << std::boolalpha << checkResult(*pAnswer, static_cast<double>(common)) << "\n";
    
    #if 0
    {   // fill with zeros
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid(32, 32);
        fill<<<blocksPerGrid, threadsPerBlock>>>(*pAnswer, 0.0);
        cudaDeviceSynchronize();
    }
    // takes 26 seconds sequentially on a CPU
    auto begin = std::chrono::high_resolution_clock::now();
    multiplyOnHost(*pAnswer, *pLHS, *pRHS);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin);
    std::cout << checkResult(*pAnswer, static_cast<double>(common)) << "\n";
    std::cout << duration.count() << "sec\n";
    #endif

    return 0;
}
