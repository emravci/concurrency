// Reference: [1] CUDA C++ Programming Guide

namespace OpenKernel { template<class Type> class Vector; }

template<class Type>
class OpenKernel::Vector
{
    public:
    using ValueType = Type;
    using SizeType = std::size_t;
    Vector(SizeType size) : data_{new ValueType[size]}, size_{size} {}
    Vector(const Vector& src) : data_{new ValueType[src.size_]}, size_{src.size_}
    {
        for(SizeType i = 0; i < size_; ++i) { data_[i] = src.data_[i]; }
    }
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&&) = delete;
    Vector& operator=(Vector&&) = delete;
    ~Vector() { delete[] data_; }
    __host__ __device__ SizeType size() const { return size_; }
    __host__ __device__ ValueType& operator[](SizeType i) { return data_[i]; }
    __host__ __device__ const ValueType& operator[](SizeType i) const { return data_[i]; }
    __host__ __device__ ValueType* begin() { return data_; }
    __host__ __device__ ValueType* end() { return &data_[size_]; }
    __host__ __device__ const ValueType* cbegin() const { return data_; }
    __host__ __device__ const ValueType* cend() const { return &data_[size_]; }
    private:
    ValueType *data_ = nullptr;
    SizeType size_ = 0;
};
