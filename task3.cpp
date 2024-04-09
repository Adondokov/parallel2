#include <iostream>
#include <cmath>
#include <omp.h>

double f(double x) {
    return x * x; // Пример функции для интегрирования
}

double integrate_simple_iteration(double a, double b, int nsteps) {
    double sum = 0.0;
    double step = (b - a) / nsteps;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < nsteps; ++i) {
        double x = a + (i + 0.5) * step;
        sum += f(x) * step;
    }

    return sum;
}

int main() {
    const int nsteps = 40000000;
    const double a = 0.0;
    const double b = 1.0;

    std::cout << "Number of Steps: " << nsteps << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Number of Threads | Time (seconds)" << std::endl;

    for (int num_threads : {1, 2, 4, 7, 8, 16, 20, 40}) {
        omp_set_num_threads(num_threads);
        double start_time = omp_get_wtime();
        double result = integrate_simple_iteration(a, b, nsteps);
        double end_time = omp_get_wtime();
        std::cout << "       " << num_threads << "        |     " << end_time - start_time << std::endl;
    }

    return 0;
}
