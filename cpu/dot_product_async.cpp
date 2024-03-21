// Reference: [1] Rainer Grimm. Concurrency with Modern C++. p:192

#include <iostream>
#include <future>
#include <array>
#include <numeric>
#include <chrono>
#include <vector>

template<std::size_t NumberOfTasks>
double dotProduct(const std::vector<double>& u, const std::vector<double>& v)
{
	std::size_t size = u.size();
	std::size_t ceiledSizePerTask = (size + NumberOfTasks - 1) / NumberOfTasks;
	std::array<std::future<double>, NumberOfTasks> futures;
	for(std::size_t t = 0; t < NumberOfTasks; ++t)
	{
		std::size_t begin = ceiledSizePerTask * t;
		std::size_t end = std::min(begin + ceiledSizePerTask, size);
		futures[t] = std::async(std::launch::async, [&u, &v, begin, end]
		{
			return std::inner_product(&u[begin], &u[end], &v[begin], 0.0);
		});
	}
	double result = 0.0;
	for(auto& future : futures) result += future.get();
	return result;
}

int main()
{
	constexpr std::size_t numberOfElements = 1024 * 1024 * 256;
	std::vector<double> u(numberOfElements, 1.0);

	// concurrent part - 93ms
	auto before = std::chrono::high_resolution_clock::now();
	double concurrent = dotProduct<10>(u, u);
	auto after = std::chrono::high_resolution_clock::now();
	std::cout << std::fixed << concurrent << " in "; 
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << "ms\n";

	// sequential part - 703ms
	before = std::chrono::high_resolution_clock::now();
	double sequential = std::inner_product(&u[0], &u[numberOfElements], &u[0], 0.0);
	after = std::chrono::high_resolution_clock::now();
	std::cout << sequential << " in ";
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << "ms\n";
	return 0;
}
