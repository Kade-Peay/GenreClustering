#include "utils.hpp"
#include <cfloat>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>

extern "C" void Malloc(Point** points, size_t size);
extern "C" void MemcpyHost(Point* devicePoints, Point* hostPoints, size_t size);
extern "C" void MemcpyDevice(Point* devicePoints, Point* hostPoints, size_t size);
extern "C" void Free(Point* points);
extern "C" float AssignToCluster(int blocks, int threadsPerBlock, Point* points, Point* centroids, int k, int numPoints);

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
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
    srand(100);

    // Initialize centroids with random points
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points.at(rand() % points.size()));
    }

    // Allocate device memory
    Point* d_points;
    Point* d_centroids;
    Malloc(&d_points, points.size());
    Malloc(&d_centroids, k);

    // Copy data to device
    MemcpyHost(d_points, points.data(), points.size());
    MemcpyHost(d_centroids, centroids.data(), k);

    int blocks = (points.size() + threadsPerBlock - 1) / threadsPerBlock;

    float cudaElapsedTime = 0;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Assign points to clusters
        cudaElapsedTime += AssignToCluster(blocks, threadsPerBlock, d_points, d_centroids, k, points.size());

        std::vector<int> nPoints(k, 0);
        std::vector<double> sumD(k, 0.0), sumV(k, 0.0), sumE(k, 0.0);

        // Accumulate points for new centroids
        MemcpyDevice(d_points, points.data(), points.size());
        for (auto& p : points) {
            int clusterId = p.cluster;
            nPoints[clusterId] += 1;
            sumD[clusterId] += p.danceability;
            sumV[clusterId] += p.valence;
            sumE[clusterId] += p.energy;
        }

        // Compute new centroids
        MemcpyDevice(d_centroids, centroids.data(), k);
        for (int clusterId = 0; clusterId < k; ++clusterId) {
            if (nPoints[clusterId] != 0) {
                centroids[clusterId].danceability = sumD[clusterId] / nPoints[clusterId];
                centroids[clusterId].valence = sumV[clusterId] / nPoints[clusterId];
                centroids[clusterId].energy = sumE[clusterId] / nPoints[clusterId];
            }
        }
        MemcpyHost(d_centroids, centroids.data(), k);
    }

    // Clean upp
    Free(d_points);
    Free(d_centroids);

    // Write results to output file
    std::ofstream myfile("shared-gpu_output.csv");
    myfile << "danceability,valence,energy,cluster\n";

    for (const auto &point : points)
    {
        myfile << point.danceability << "," << point.valence << "," << point.energy << "," << point.cluster << "\n";
    }
    myfile.close();

    std::cout << "Clustering complete. Results saved to shared-gpu_output.csv\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    std::cout << "CUDA elapsed time: " << cudaElapsedTime << " milliseconds" << std::endl;
    return 0;
}
