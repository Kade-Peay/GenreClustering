#include "utils.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

__global__ void AssignToCluster(Point* points, Point* centroids, int k, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    Point p = points[idx];
    p.minDist = DBL_MAX;

    for (int clusterId = 0; clusterId < k; clusterId++)
    {
        double dist = 
            (p.danceability - centroids[clusterId].danceability) * (p.danceability - centroids[clusterId].danceability) +
            (p.valence - centroids[clusterId].valence) * (p.valence - centroids[clusterId].valence) + 
            (p.energy - centroids[clusterId].energy) * (p.energy - centroids[clusterId].energy);

        if (dist < p.minDist)
        {
            p.minDist = dist;
            p.cluster = clusterId;
        }
    }
    points[idx] = p;
}

extern "C" void Malloc(Point** points, size_t size){
    cudaError_t err = cudaMalloc(points, size * sizeof(Point));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

extern "C" void MemcpyHost(Point* devicePoints, Point* hostPoints, size_t size){
    cudaError_t err = cudaMemcpy(devicePoints, hostPoints, size * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

extern "C" void MemcpyDevice(Point* devicePoints, Point* hostPoints, size_t size){
    cudaError_t err = cudaMemcpy(hostPoints, devicePoints, size * sizeof(Point), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

extern "C" void Free(Point* points){
    cudaError_t err = cudaFree(points);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

extern "C" float AssignToCluster(int blocks, int threadsPerBlock, Point* points, Point* centroids, int k, int numPoints) {
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    AssignToCluster<<<blocks, threadsPerBlock>>>(points, centroids, k, numPoints);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    return milliseconds;
}
