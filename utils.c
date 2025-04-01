#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

double **read_csv(const char *filename, int *n, int *d)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error opening file %s: %s\n", filename, strerror(errno));
        return NULL;
    }

    char buffer[4096];

    // Skip header with error checking
    if (!fgets(buffer, sizeof(buffer), file))
    {
        fprintf(stderr, "Empty file or read error\n");
        fclose(file);
        return NULL;
    }

    // First pass to count rows and features
    int num_features = 0;
    int num_points = 0;
    long file_pos = ftell(file); // Remember position after header

    while (fgets(buffer, sizeof(buffer), file))
    {
        num_points++;
        if (num_points == 1)
        {
            char temp_buffer[4096];
            strcpy(temp_buffer, buffer);
            char *token = strtok(temp_buffer, ",");
            while (token)
            {
                num_features++;
                token = strtok(NULL, ",");
            }
        }
    }

    if (num_points == 0)
    {
        fclose(file);
        return NULL;
    }

    *n = num_points;
    *d = num_features;

    // Allocate memory
    double **data = malloc(*n * sizeof(double *));
    if (!data)
    {
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < *n; i++)
    {
        data[i] = malloc(*d * sizeof(double));
        if (!data[i])
        {
            for (int j = 0; j < i; j++)
                free(data[j]);
            free(data);
            fclose(file);
            return NULL;
        }
    }

    // Second pass to read data
    fseek(file, file_pos, SEEK_SET);
    for (int i = 0; i < *n; i++)
    {
        if (!fgets(buffer, sizeof(buffer), file))
        {
            // Handle read error
            for (int j = 0; j <= i; j++)
                free(data[j]);
            free(data);
            fclose(file);
            return NULL;
        }

        char *token = strtok(buffer, ",");
        for (int j = 0; j < *d && token; j++)
        {
            data[i][j] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
    return data;
}

double **init_centroids(double **data, int n, int d, int k)
{
    double **centroids = malloc(k * sizeof(double *));
    if (!centroids)
        return NULL;

    srand(time(NULL));

    for (int i = 0; i < k; i++)
    {
        centroids[i] = malloc(d * sizeof(double));
        if (!centroids[i])
        {
            for (int j = 0; j < i; j++)
                free(centroids[j]);
            free(centroids);
            return NULL;
        }

        int random_index = rand() % n;
        memcpy(centroids[i], data[random_index], d * sizeof(double));
    }

    return centroids;
}

int find_nearest_centroid(double *point, double **centroids, int k, int d)
{
    int nearest = 0;
    double min_dist = __DBL_MAX__;

    for (int i = 0; i < k; i++)
    {
        double dist = 0.0;
        for (int j = 0; j < d; j++)
        {
            double diff = point[j] - centroids[i][j];
            dist += diff * diff;
        }
        if (dist < min_dist)
        {
            min_dist = dist;
            nearest = i;
        }
    }

    return nearest;
}

void free_data(double **data, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(data[i]);
    }
    free(data);
}
