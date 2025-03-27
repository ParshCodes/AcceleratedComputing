#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

double f(double x) {
    return acos(cos(x) / (1.0 + 2.0 * cos(x)));
}

double simpsons_rule(double a, double b, int n, int num_threads) {
    if (n % 2 != 0) {
        cout << "Error: n must be even for Simpson's rule." << endl;
        return 0.0;
    }

    double h = (b - a) / n;
    double integral = f(a) + f(b);
    double y1 = 0.0, y2 = 0.0;

    #pragma omp parallel for reduction(+:y1) num_threads(num_threads)
    for (int i = 1; i < n; i += 2) {
        y1 += f(a + i * h);
    }

    #pragma omp parallel for reduction(+:y2) num_threads(num_threads)
    for (int i = 2; i < n; i += 2) {
        y2 += f(a + i * h);
    }

    integral += 4.0 * y1 + 2.0 * y2;
    integral *= h / 3.0;

    return integral;
}

int main() {
    double a = 0.0, b = M_PI / 2.0; 
    int n;  
    const double exact_value = (5.0 * M_PI * M_PI) / 24.0; 
    cin >>n;
    cout << "n = " << n << endl;

    for (int exp = 0; exp <= 7; exp++) {
        int num_threads = pow(2, exp);
        omp_set_num_threads(num_threads);

        double start_time = omp_get_wtime();
        double result = simpsons_rule(a, b, n, num_threads);
        double end_time = omp_get_wtime();

        double error = fabs(result - exact_value);
        double elapsed_time = end_time - start_time;

        cout << "Threads: " << num_threads << ", Error: " << scientific << error
             << ", Time: " << elapsed_time << " seconds" << endl;
    }

    return 0;
}
