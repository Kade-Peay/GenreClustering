#include <mpi.h>
#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>

/*
    Author: Rebecca Lamoreaux
*/

// We need all points so we can creat the centroids correctly.
// Only access allPoints if world_rank == 0
void kMeansClustering(std::vector<Point> *localPoints, std::vector<Point> *allPoints, int epochs, int k, int world_rank)
{
    //Set seed for reproducibility
    std::vector<Point> centroids;

    // Initialize centroids randomly and broadcast them (By root)
    if (world_rank == 0){
        srand(100);
        for (int i = 0; i < k; ++i)
        {
            centroids.push_back(allPoints->at(rand() % allPoints->size()));
        }
    }
    else {
        centroids.resize(k); // Allocate memory for broadcast on other ranks
    }
    MPI_Bcast(centroids.data(), k * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Parallel assignment of points to clusters
        #pragma omp parallel for
        for (size_t i = 0; i < localPoints->size(); ++i) {
            Point &p = (*localPoints)[i];
            p.minDist = DBL_MAX;
            
            for (int clusterId = 0; clusterId < k; ++clusterId) {
                double dist = centroids[clusterId].distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
            }
        }

        // Set up local gathering of stats for all clusters
        int local_nPoints[k] = {0};
        double local_sumD[k] = {0.0};
        double local_sumV[k] = {0.0};
        double local_sumE[k] = {0.0};

        // Parallel accumulation of points
        #pragma omp parallel for reduction(+:local_nPoints, local_sumD, local_sumV, local_sumE)
        for (size_t i = 0; i < localPoints->size(); ++i) {
            Point &p = (*localPoints)[i];
            int clusterId = p.cluster;
            local_nPoints[clusterId] += 1;
            local_sumD[clusterId] += p.danceability;
            local_sumV[clusterId] += p.valence;
            local_sumE[clusterId] += p.energy;
            p.minDist = DBL_MAX;  // Reset for next iteration
        }
        // Redcue thread points into global sums
        int global_nPoints[k] = {0};
        double global_sumD[k] = {0.0};
        double global_sumV[k] = {0.0};
        double global_sumE[k] = {0.0};

        MPI_Allreduce(local_nPoints, global_nPoints, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sumD, global_sumD, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sumV, global_sumV, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sumE, global_sumE, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update centroids and watch for convergence (ONLY ROOT)
        bool localConverged = true;

        if (world_rank == 0){
            std::vector<Point> newCentroids(k);
            for (int clusterId = 0; clusterId < k; ++clusterId)
            {
                if (global_nPoints[clusterId] == 0) continue;

                newCentroids[clusterId].danceability = global_sumD[clusterId] / global_nPoints[clusterId];
                newCentroids[clusterId].valence      = global_sumV[clusterId] / global_nPoints[clusterId];
                newCentroids[clusterId].energy       = global_sumE[clusterId] / global_nPoints[clusterId];
                
                double delta = centroids[clusterId].distance(newCentroids[clusterId]);
                if (delta > convergenceDelta){
                    localConverged = false;
                }
            }
            centroids = newCentroids;
        }

        // Broadcast updated centroids
        MPI_Bcast(centroids.data(), k * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Broadcast convergence flag
        int convergedInt = localConverged ? 1 : 0;
        MPI_Bcast(&convergedInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (convergedInt == 1) {
            if (world_rank == 0) {
                std::cout << "Converged at epoch " << epoch << std::endl;
            }
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    // Set up MPI for Distribution stuff
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // first check for proper command line args
    if (argc != 4) {
        if (world_rank == 0)
            std::cerr << "Usage: " << argv[0] << " <input_file> <number_of_clusters> <thread_count>\n";
        MPI_Finalize();
        return 1;
    }
    
    // get thread num from args
    std::string inputFile;
    int k, threads, totalSize;
    int epochs = 100; // number of iterations

    std::vector<Point> allPoints;
    if (world_rank == 0){
        inputFile = argv[1];
        k = std::stoi(argv[2]);
        threads = std::stoi(argv[3]);
        allPoints = readcsv(inputFile);
        totalSize = allPoints.size();
        
        if (allPoints.empty())
        {
            std::cerr << "No data points loaded. Check your input file.\n";
            MPI_Finalize();
            return 1;
        }
    }
    
    // Broadcast size to all
    MPI_Bcast(&totalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Split data evenly among processes
    int localSize = totalSize / world_size;
    std::vector<Point> localPoints(localSize);
    MPI_Scatter(allPoints.data(), localSize * sizeof(Point), MPI_BYTE,
                localPoints.data(), localSize * sizeof(Point), MPI_BYTE,
                0, MPI_COMM_WORLD);

    omp_set_num_threads(threads);
    kMeansClustering(&localPoints, &allPoints, epochs, k, world_rank);

    // Gather all points at root
    MPI_Gather(localPoints.data(), localSize * sizeof(Point), MPI_BYTE,
               allPoints.data(), localSize * sizeof(Point), MPI_BYTE,
               0, MPI_COMM_WORLD);

    // Write results to output file
    if (world_rank == 0) {
        std::ofstream myfile("distributed_output.csv");
        myfile << "danceability,valence,energy,cluster\n";
        for (const auto &point : allPoints)
            myfile << point.danceability << "," << point.valence << "," << point.energy << "," << point.cluster << "\n";
        myfile.close();
        std::cout << "Results saved to distributed_output.csv\n";
    }

    MPI_Finalize();
    return 0;
}
