#include "utils.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

__global__ void AssignToCluster(Point* points, Point* centroids, int clusterId) {
    int idx = threadIdx.x;

    Point p = points[idx];

    double dist = 
        (p.danceability - centroids[clusterId].danceability) * (p.danceability - centroids[clusterId].danceability) +
        (p.valence - centroids[clusterId].valence) * (p.valence - centroids[clusterId].valence) + 
        (p.energy - centroids[clusterId].energy) * (p.energy - centroids[clusterId].energy);

    if (dist < p.minDist)
    {
        p.minDist = dist;
        p.cluster = clusterId;
    }
    points[idx] = p;
}

__global__ void ResetDistance(Point* points){
    int idx = threadIdx.x;
    points[idx].minDist = DBL_MAX;
}

extern "C" void Malloc(Point** points, int size){
    cudaMalloc(points, size);
}

extern "C" void MemcpyHost(Point* devicePoints, Point* hostPoints, int size){
    cudaMemcpy(devicePoints, hostPoints, size * sizeof(Point), cudaMemcpyHostToDevice);
}

extern "C" void MemcpyDevice(Point* devicePoints, Point* hostPoints, int size){
    cudaMemcpy(hostPoints, devicePoints, size * sizeof(Point), cudaMemcpyDeviceToHost);

}

extern "C" void Synchronize(){
    cudaDeviceSynchronize();
}

extern "C" void Free(Point* points){
    cudaFree(points);
}

extern "C" void AssignToCluster(int blocks, int threadsPerBlock, Point* points, Point* centroids, int centroidId) {
    AssignToCluster<<<blocks, threadsPerBlock>>>(points, centroids, centroidId);
}

extern "C" void ResetDistance(int blocks, int threadsPerBlock, Point* points) {
    ResetDistance<<<blocks, threadsPerBlock>>>(points);
}
