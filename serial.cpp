#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>

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

        // Compute new centroids
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
