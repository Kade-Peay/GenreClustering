#ifndef UTILS_H
#define UTILS_H

double** read_csv(const char* filename, int* n, int* d);
double** init_centroids(double** data, int n, int d, int k);
int find_nearest_centroid(double* point, double** centroids, int k, int d);
void free_data(double** data, int n);

#endif
