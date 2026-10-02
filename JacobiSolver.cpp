#include "JacobiSolver.h"

JacobiSolver::JacobiSolver(const std::vector<std::vector<double>>& _A, const std::vector<double>& _b)
{
    A = _A;
    b = _b;

    n = b.size();

    x.resize(n, 0.0);
}


bool JacobiSolver::solve(double tolerance, int maxIterations)
{
    std::vector<double> x_new(n);

    double startTime = omp_get_wtime();

    int iteration = 0;
    double error = 0.0;

    while (iteration < maxIterations)
    {

#pragma omp parallel for

        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;

            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    sum += A[i][j] * x[j];
                }
            }

            x_new[i] = (b[i] - sum) / A[i][i];
        }


        double squaredError = 0.0;

#pragma omp parallel for reduction(+:squaredError)
        for (int i = 0; i < n; i++)
        {
            double difference = x_new[i] - x[i];

            squaredError += difference * difference;
        }

        error = std::sqrt(squaredError);


#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            x[i] = x_new[i];
        }


        iteration++;


        if (error < tolerance)
        {
            break;
        }
    }


    double endTime = omp_get_wtime();

    std::cout << "Количество итераций: " << iteration << std::endl;

    std::cout << "Ошибка: " << error << std::endl;

    std::cout << "Время расчета: " << endTime - startTime << " секунд" << std::endl;


    return error < tolerance;
}


const std::vector<double>& JacobiSolver::getSolution() const
{
    return x;
}


void JacobiSolver::printSolution() const
{
    for (int i = 0; i < n; i++)
    {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
}
