#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>

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
        // Parallel assignment of points to clusters
        #pragma omp parallel for 
        for (size_t i = 0; i < points->size(); ++i)
        {
            Point &p = (*points)[i];
            p.minDist = DBL_MAX;

            for (int clusterId = 0; clusterId < k; ++clusterId)
            {
                double dist = centroids[clusterId].distance(p);
                if (dist < p.minDist)
                {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
            }
        }
    }

    // Reduction variables for accumulation
    std::vector<int> nPoints(k, 0);
    std::vector<double> sumD(k, 0.0), sumV(k,0.0), sumE(k,0.0);

    // Parallel accumulation of points
    #pragma omp parallel for reduction(+:nPoints, sumD, sumV, sumE)
    for (size_t i = 0; i < points->size(); ++i)
    {
        const Point &p = (*points)[i];
        int clusterId = p.cluster;
        nPoints[clusterId] += 1;
        sumD[clusterId] += p.danceability;
        sumV[clusterId] += p.valence;
        sumE[clusterId] += p.energy;
    }

    // compute the new centroids
    // done serially still
    for (int clusterId = 0; clusterId < k; ++clusterId)
    {
        if (nPoints[clusterId] != 0) 
        {
            centroids[clusterId].danceability = sumD[clusterId] / nPoints[clusterId];
            centroids[clusterId].valence = sumV[clusterId] / nPoints[clusterId];
            centroids[clusterId].energy = sumE[clusterId] / nPoints[clusterId];
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

    if (points.empty())
    {
        std::cerr << "No data points loaded. Check your input file.\n";
        return 1;
    }

    int k = 6;       // number of clusters
    int epochs = 100; // number of iterations


    
    // set number of threads
    omp_set_num_threads(threads);

    kMeansClustering(&points, epochs, k);

    // Write results to output file
    std::ofstream myfile("output.csv");
    myfile << "danceability,valence,energy,cluster\n";

    for (const auto &point : points)
    {
        myfile << point.danceability << "," << point.valence << "," << point.energy << "," << point.cluster << "\n";
    }
    myfile.close();

    std::cout << "Clustering complete. Results saved to output.csv\n";
    return 0;
}
