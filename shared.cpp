#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <chrono>

/*
    Author: Kade Peay
*/

void kMeansClustering(std::vector<Point> *points, int epochs, int k)
{
    std::vector<Point> centroids;
    srand(100);

    // Initialize centroids with random points
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points->at(rand() % points->size()));
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Parallel assignment of points to clusters
        #pragma omp parallel for
        for (size_t i = 0; i < points->size(); ++i) {
            Point &p = (*points)[i];
            p.minDist = DBL_MAX;
            
            for (int clusterId = 0; clusterId < k; ++clusterId) {
                double dist = centroids[clusterId].distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
            }
        }

        // Reduction variables for accumulation
        int nPoints[k] = {0};
        double sumD[k] = {0.0};
        double sumV[k] = {0.0};
        double sumE[k] = {0.0};

        // Parallel accumulation of points
        #pragma omp parallel for reduction(+:nPoints, sumD, sumV, sumE)
        for (size_t i = 0; i < points->size(); ++i) {
            Point &p = (*points)[i];
            int clusterId = p.cluster;
            nPoints[clusterId] += 1;
            sumD[clusterId] += p.danceability;
            sumV[clusterId] += p.valence;
            sumE[clusterId] += p.energy;
            p.minDist = DBL_MAX;  // Reset for next iteration
        }

        // Update centroids
        for (int clusterId = 0; clusterId < k; ++clusterId) {
            if (nPoints[clusterId] != 0) {
                centroids[clusterId].danceability = sumD[clusterId] / nPoints[clusterId];
                centroids[clusterId].valence = sumV[clusterId] / nPoints[clusterId];
                centroids[clusterId].energy = sumE[clusterId] / nPoints[clusterId];
            }
        }
    }
}
int main(int argc, char *argv[])
{
    // first check for proper command line args
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <number_of_clusters> <thread_count>" << std::endl;
        return 1;
    }
    
    // get thread num from args
    std::string inputFile = argv[1];
    int k = std::stoi(argv[2]);
    int threads = atoi(argv[3]);
    std::vector<Point> points = readcsv(inputFile);
    int epochs = 100; // number of iterations

    if (points.empty())
    {
        std::cerr << "No data points loaded. Check your input file.\n";
        return 1;
    }
    
    // set number of threads
    omp_set_num_threads(threads);

    // start timer
    const auto start = std::chrono::high_resolution_clock::now();

    // run clustering
    kMeansClustering(&points, epochs, k);

    // end timer
    const auto end = std::chrono::high_resolution_clock::now();

    // calculate time taken
    const std::chrono::duration<double> timeTaken = end - start;

    // Write results to output file
    std::ofstream myfile("shared_output.csv");
    myfile << "danceability,valence,energy,cluster\n";

    for (const auto &point : points)
    {
        myfile << point.danceability << "," << point.valence << "," << point.energy << "," << point.cluster << "\n";
    }
    myfile.close();

    // Report the file being written 
    std::cout << "Clustering complete. Results saved to shared_output.csv\n";

    // Report the time taken calculated earlier
    std::cout << "Time Taken: " << timeTaken.count() << " seconds" << std::endl;
    return 0;
}
