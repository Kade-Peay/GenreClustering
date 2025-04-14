#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>
#include <cuda_runtime.h>

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
};

std::vector<Point> readcsv()
{
    std::vector<Point> points;
    std::string line;

    // pick to use full file or sample file
    // std::ifstream file("tracks_features.csv");
    std::ifstream file("sample_data.csv");

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;

        bool inQuotes = false;
        std::string current;

        // handle quoted fields 
        for (char c : line) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                row.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        row.push_back(current); // add last field

        // Check if there is enough columns 
        if (row.size() >= 11) {
            try {
                size_t pos;
                double x = std::stod(row[9], &pos);
                if(pos != row[9].length()) continue; // skip if not converted
                double y = std::stod(row[10], &pos);
                if (pos != row[10].length()) continue;
                
                points.push_back(Point(x, y));
            } catch (const std::exception& e) {
                std::cerr << "Skipping line due to parse error: " << line << "\n";
                continue;
            }
        }
    }
    return points;
}

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

    // Launch kernel (1 block, k threads or more depending on your needs)
    int threadsPerBlock = 256;
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
