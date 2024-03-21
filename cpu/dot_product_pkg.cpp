// Reference: [1] Rainer Grimm. Concurrency with Modern C++. p:194

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
	for(int t = 0; t < NumberOfTasks; ++t)
	{
		std::packaged_task task(std::inner_product<const double*, const double*, double>);
		futures[t] = task.get_future();
		std::size_t begin = t * ceiledSizePerTask;
		std::size_t end = std::min(begin + ceiledSizePerTask, size);
		std::thread thread(std::move(task), &u[begin], &u[end], &v[begin], 0.0);
		thread.detach();
	}
	double result = 0.0;
	for(auto& future : futures) result += future.get();
	return result;
}

int main()
{
	constexpr std::size_t numberOfElements = 1024 * 1024 * 256;
	std::vector<double> u(numberOfElements, 1.0);

	// concurrent - 89ms
	auto before = std::chrono::high_resolution_clock::now();
	double concurrent = dotProduct<10>(u, u);
	auto after = std::chrono::high_resolution_clock::now();
	std::cout << std::fixed << concurrent << " in ";
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << "ms\n";

	// sequential - 708ms
	before = std::chrono::high_resolution_clock::now();
	double sequential = std::inner_product(&u[0], &u[numberOfElements], &u[0], 0.0);
	after = std::chrono::high_resolution_clock::now();
	std::cout << sequential << " in ";
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << "ms\n";
	return 0;
}
