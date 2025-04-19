#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>

extern "C" void Malloc(Point** points, int size);
extern "C" void MemcpyHost(Point* devicePoints, Point* hostPoints, int size);
extern "C" void MemcpyDevice(Point* devicePoints, Point* hostPoints, int size);
extern "C" void Synchronize();
extern "C" void Free(Point* points);
extern "C" void AssignToCluster(int blocks, int threadsPerBlock, Point* points, Point* centroids, int centroidId);
extern "C" void ResetDistance(int blocks, int threadsPerBlock, Point* points);

int main(int argc, char *argv[])
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
    srand(100);

    // Initialize centroids with random points
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points.at(rand() % points.size()));
    }

    // Allocate device memory
    Point* d_points;
    Point* d_centroids;
    Malloc(&d_points, points.size() * sizeof(Point));
    Malloc(&d_centroids, k * sizeof(Point));

    // Copy data to device
    MemcpyHost(d_points, points.data(), points.size());
    MemcpyHost(d_centroids, centroids.data(), k);

    int blocks = (k + threadsPerBlock - 1) / threadsPerBlock;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Assign points to clusters
        for (size_t i = 0; i < centroids.size(); ++i)
        {
            AssignToCluster(blocks, threadsPerBlock, d_points, d_centroids, i);
        }
        Synchronize();

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

        // reset distance
        ResetDistance(blocks, threadsPerBlock, d_points);
        Synchronize();

        // Compute new centroids
        MemcpyDevice(d_centroids, centroids.data(), k);
        for (auto c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            if (nPoints[clusterId] != 0)
            {
                c->danceability = sumD[clusterId] / nPoints[clusterId];
                c->valence = sumV[clusterId] / nPoints[clusterId];
                c->energy = sumE[clusterId] / nPoints[clusterId];
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
    return 0;
}
