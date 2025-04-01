#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "utils.h"

#define MAX_ITER 100
#define THRESHOLD 1e-5

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <data_file> <k> <num_threads>\n", argv[0]);
        return 1;
    }

    // Initialize random seed for centroid selection
    srand(time(NULL));

    char *data_file = argv[1];
    int k = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    omp_set_num_threads(num_threads);

    int n, d;
    double **data = read_csv(data_file, &n, &d);
    if (!data)
    {
        fprintf(stderr, "Failed to read data file\n");
        return 1;
    }

    double **centroids = init_centroids(data, n, d, k);
    double **new_centroids = malloc(k * sizeof(double *));
    for (int i = 0; i < k; i++)
    {
        new_centroids[i] = malloc(d * sizeof(double));
    }

    int *labels = malloc(n * sizeof(int));
    int iter = 0;
    double max_diff;

    do
    {
        max_diff = 0.0;

// Assignment step
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            labels[i] = find_nearest_centroid(data[i], centroids, k, d);
        }

        // Update step
        double **sums = malloc(k * sizeof(double *));
        int *counts = calloc(k, sizeof(int));
        for (int i = 0; i < k; i++)
        {
            sums[i] = calloc(d, sizeof(double));
        }

#pragma omp parallel for reduction(+ : counts[ : k])
        for (int i = 0; i < n; i++)
        {
            int cluster = labels[i];
            counts[cluster]++;
            for (int j = 0; j < d; j++)
            {
                sums[cluster][j] += data[i][j];
            }
        }

        // Calculate new centroids
        for (int c = 0; c < k; c++)
        {
            if (counts[c] == 0)
                continue;
            for (int j = 0; j < d; j++)
            {
                new_centroids[c][j] = sums[c][j] / counts[c];
                double diff = fabs(new_centroids[c][j] - centroids[c][j]);
                if (diff > max_diff)
                    max_diff = diff;
            }
        }

        // Copy new centroids
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                centroids[c][j] = new_centroids[c][j];
            }
        }

        // Free temporary memory
        for (int c = 0; c < k; c++)
            free(sums[c]);
        free(sums);
        free(counts);
    } while (max_diff > THRESHOLD && ++iter < MAX_ITER);

    printf("Converged in %d iterations\n", iter);
    for (int c = 0; c < k; c++)
    {
        printf("Centroid %d: [", c);
        for (int j = 0; j < d; j++)
        {
            printf("%.3f%s", centroids[c][j], j < d - 1 ? ", " : "");
        }
        printf("]\n");
    }

    // Cleanup
    free(labels);
    free_data(data, n);
    for (int c = 0; c < k; c++)
    {
        free(centroids[c]);
        free(new_centroids[c]);
    }
    free(centroids);
    free(new_centroids);

    return 0;
}
