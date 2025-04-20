#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <chrono>
#include <iostream>
#include <numeric>
#include <iomanip>

/*
    Author: Kade Peay
*/

void kMeansClustering(std::vector<Point> *points, int epochs, int k)
{
    std::vector<Point> centroids;
    srand(100);

    // Initialize centroids with random points
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points->at(rand() % points->size()));
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Assign points to clusters
        for (auto c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);

            for (auto it = points->begin(); it != points->end(); ++it)
            {
                Point p = *it;
                double dist = c->distance(p);
                if (dist < p.minDist)
                {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
                *it = p;
            }
        }

        std::vector<int> nPoints(k, 0);
        std::vector<double> sumD(k, 0.0), sumV(k, 0.0), sumE(k, 0.0);

        // Accumulate points for new centroids
        for (auto& p : *points) {
            int clusterId = p.cluster;
            nPoints[clusterId] += 1;
            sumD[clusterId] += p.danceability;
            sumV[clusterId] += p.valence;
            sumE[clusterId] += p.energy;
            p.minDist = DBL_MAX; // reset distance
        }

        // Update centroids and watch for convergence
        bool converged = true;
        std::vector<Point> newCentroids(k);
        for (int clusterId = 0; clusterId < k; ++clusterId)
        {
            if (nPoints[clusterId] == 0) continue;

            newCentroids[clusterId].danceability = sumD[clusterId] / nPoints[clusterId];
            newCentroids[clusterId].valence      = sumV[clusterId] / nPoints[clusterId];
            newCentroids[clusterId].energy       = sumE[clusterId] / nPoints[clusterId];
            
            double delta = centroids[clusterId].distance(newCentroids[clusterId]);
            if (delta > convergenceDelta){
                converged = false;
            }
        }
        centroids = newCentroids;

        if (converged){
            std::cout << "Converged at epoch " << epoch << std::endl;
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <number_of_clusters>" << std::endl;
        return -1;
    }

    std::string inputFile = argv[1];
    int k = std::stoi(argv[2]);
    std::vector<Point> points = readcsv(inputFile);

    if (points.empty())
    {
        std::cerr << "No data points loaded. Check your input file.\n";
        return 1;
    }

    int epochs = 100; // number of iterations

    // start timer
    const auto start = std::chrono::high_resolution_clock::now();

    // call clustering
    kMeansClustering(&points, epochs, k); 
        
    // end timing
    const auto end = std::chrono::high_resolution_clock::now();

    // calculate time taken
    const std::chrono::duration<double> timeTaken = end - start;
    

    // Write results to output file
    std::ofstream myfile("serial_output.csv");
    myfile << "danceability,valence,energy,cluster\n";

    for (const auto &point : points)
    {
        myfile << point.danceability << "," << point.valence << "," << point.energy << "," << point.cluster << "\n";
    }
    myfile.close();

    // Report sucessful output
    std::cout << "Clustering complete. Results saved to serial_output.csv\n";

    // Report the time calcualted earlier
    std::cout << "Time taken: " << timeTaken.count() << " seconds." << std::endl;
    
    return 0;
}
