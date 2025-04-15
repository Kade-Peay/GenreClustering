#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>

void kMeansClustering(std::vector<Point> *points, int epochs, int k)
{
    std::vector<Point> centroids;
    srand(time(0));

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
        std::vector<double> sumX(k, 0.0);
        std::vector<double> sumY(k, 0.0);

        // Accumulate points for new centroids
        for (auto it = points->begin(); it != points->end(); ++it)
        {
            int clusterId = it->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += it->x;
            sumY[clusterId] += it->y;
            it->minDist = DBL_MAX; // reset distance
        }

        // Compute new centroids
        for (auto c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            if (nPoints[clusterId] != 0)
            {
                c->x = sumX[clusterId] / nPoints[clusterId];
                c->y = sumY[clusterId] / nPoints[clusterId];
            }
        }
    }
}

int main()
{
    std::vector<Point> points = readcsv();

    if (points.empty())
    {
        std::cerr << "No data points loaded. Check your input file.\n";
        return 1;
    }

    int k = 6;       // number of clusters
    int epochs = 100; // number of iterations
    kMeansClustering(&points, epochs, k);

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
