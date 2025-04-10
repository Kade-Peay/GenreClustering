#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>

struct Point
{
    double x, y; // coordinates (using danceability and energy)
    int cluster;
    double minDist;

    Point() : x(0.0),
              y(0.0),
              cluster(-1),
              minDist(DBL_MAX) {}

    Point(double x, double y) : x(x),
                                y(y),
                                cluster(-1),
                                minDist(DBL_MAX) {}

    double distance(Point p)
    {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

std::vector<Point> readcsv()
{
    std::vector<Point> points;
    std::string line;
    // std::ifstream file("tracks_features.csv");
    std::ifstream file("sample_data.csv");

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;

        // Split line into cells
        while (std::getline(lineStream, cell, ','))
        {
            row.push_back(cell);
        }

        // Check if we have enough columns
        if (row.size() >= 10)
        {
            try
            {
                // Using danceability (column 9) and energy (column 10) as x,y coordinates
                double x = std::stod(row[9]);  // danceability
                double y = std::stod(row[10]); // energy
                points.push_back(Point(x, y));
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing line: " << line << "\n";
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }
    return points;
}

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

    int k = 3;       // number of clusters
    int epochs = 10; // number of iterations
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