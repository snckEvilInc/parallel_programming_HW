#define NOMINMAX
#include <windows.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include "JacobiSolver.h"


std::vector<std::vector<double>>createMatrix(int n)
{
    std::vector<std::vector<double>> A(n, std::vector<double>(n));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                A[i][j] = n + 1;
            }
            else
            {
                A[i][j] = 1;
            }
        }
    }

    return A;
}


// Точное решение: x[i] = 2
// Для такой матрицы b[i] = 4 * n

std::vector<double>
createVector(int n)
{
    std::vector<double> b(n);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        b[i] = 4.0 * n;
    }

    return b;
}


int main()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);


    int numThreads = 7;
    omp_set_num_threads(numThreads);


#pragma omp parallel
    {
#pragma omp single
        {
            std::cout << "Количество потоков: " << omp_get_num_threads() << std::endl;
        }
    }


    int n = 1024;


    double tolerance = 1e-6;
    int maxIterations = 20000;


    std::cout << std::endl;
    std::cout << "Метод Якоби + OpenMP" << std::endl;

    std::cout << "Размер системы: " << n << " x " << n << std::endl;

    std::cout << "Точность: " << tolerance << std::endl;

    std::cout << "Максимальное количество итераций: " << maxIterations << std::endl;

    std::cout << std::endl;
    std::cout << "Создание матрицы A..." << std::endl;

    std::vector<std::vector<double>> A = createMatrix(n);


    std::cout << "Создание вектора b..." << std::endl;

    std::vector<double> b = createVector(n);


    JacobiSolver solver(A, b);


    std::cout << std::endl;
    std::cout << "Начало расчета..." << std::endl;

    bool result = solver.solve(tolerance, maxIterations);


    if (result)
    {
        std::cout << "Метод сошелся." << std::endl;
    }
    else
    {
        std::cout << "Метод не сошелся." << std::endl;
    }


    const std::vector<double>& x = solver.getSolution();


    std::cout << std::endl;
    std::cout << "Первые 10 элементов решения:" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << "x[" << i << "] = " << std::setprecision(10) << x[i] << std::endl;
    }


    return 0;
}
