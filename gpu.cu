#include "utils.hpp"
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>
#include <cuda_runtime.h>

__global__ void AssignToCluster(Point* points, Point* centroids, int clusterId) {
    int idx = threadIdx.x;

    Point p = points[idx];

    double dist = (p.x - centroids[clusterId].x) * (p.x - centroids[clusterId].x) +
                  (p.y - centroids[clusterId].y) * (p.y - centroids[clusterId].y);

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

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <number_of_clusters> <threads_per_block>" << std::endl;
        return -1;
    }

    std::string inputFile = argv[1];
    int k = std::stoi(argv[2]);
    int threadsPerBlock = std::stoi(argv[3]);

    std::vector<Point> points = readcsv(inputFile);

    if (points.empty())
    {
        std::cerr << "No data points loaded. Check your input file.\n";
        return 1;
    }

    int epochs = 100; // number of iterations

    std::vector<Point> centroids;
    srand(time(0));

    // Initialize centroids with random points
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points.at(rand() % points.size()));
    }

    // Allocate device memory
    Point* d_points;
    Point* d_centroids;
    cudaMalloc(&d_points, points.size() * sizeof(Point));
    cudaMalloc(&d_centroids, k * sizeof(Point));

    // Copy data to device
    cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Point), cudaMemcpyHostToDevice);

    int blocks = (k + threadsPerBlock - 1) / threadsPerBlock;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Assign points to clusters
        for (int i = 0; i < centroids.size(); ++i)
        {
            AssignToCluster<<<blocks, threadsPerBlock>>>(d_points, d_centroids, i);
        }
        cudaDeviceSynchronize();

        std::vector<int> nPoints(k, 0);
        std::vector<double> sumX(k, 0.0);
        std::vector<double> sumY(k, 0.0);

        // Accumulate points for new centroids
        cudaMemcpy(points.data(), d_points, points.size() * sizeof(Point), cudaMemcpyDeviceToHost);
        for (auto it = points.begin(); it != points.end(); ++it)
        {
            int clusterId = it->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += it->x;
            sumY[clusterId] += it->y;
        }

        // reset distance
        ResetDistance<<<blocks, threadsPerBlock>>>(d_points);
        cudaDeviceSynchronize();

        // Compute new centroids
        cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Point), cudaMemcpyDeviceToHost);
        for (auto c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            if (nPoints[clusterId] != 0)
            {
                c->x = sumX[clusterId] / nPoints[clusterId];
                c->y = sumY[clusterId] / nPoints[clusterId];
            }
        }
        cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Point), cudaMemcpyHostToDevice);
    }

    // Clean up
    cudaFree(d_points);
    cudaFree(d_centroids);

    // Write results to output file
    std::ofstream myfile("output.csv");
    myfile << "danceability,energy,cluster\n";

    for (const auto &point : points)
    {
        myfile << point.x << "," << point.y << "," << point.cluster << "\n";
    }
    myfile.close();

    std::cout << "Clustering complete. Results saved to output.csv\n";
    return 0;
}
