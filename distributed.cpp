#include <mpi.h>
#include "utils.hpp"
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <chrono>

/*
    Author: Rebecca Lamoreaux
*/

void kMeansClustering(std::vector<Point> &localPoints, int epochs, int k, int world_rank, int world_size)
{
    //Set seed for reproducibility
    std::vector<Point> centroids;

    // Initialize centroids randomly and broadcast them (By root)
    if (world_rank == 0){
        srand(100);

        for (int i = 0; i < k; ++i)
        {
            centroids.push_back(localPoints[rand() % localPoints.size()]);
        }

    }
    MPI_Bcast(centroids.data(), k * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Assign local points to nearest centroid
        for (Point &p: localPoints)
        {
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
    

        // Set up local gathering of stats for all clusters
        std::vector<int> local_nPoints(k, 0);
        std::vector<double> local_sumD(k, 0.0), local_sumV(k, 0.0), local_sumE(k, 0.0);

        for (const Point &p : localPoints)
        {
            int clusterId = p.cluster;
            local_nPoints[clusterId]++;
            local_sumD[clusterId] += p.danceability;
            local_sumV[clusterId] += p.valence;
            local_sumE[clusterId] += p.energy;
        }

        // Redcue thread points into global sums
        std::vector<int> global_nPoints(k);
        std::vector<double> global_sumD(k), global_sumV(k), global_sumE(k);

        MPI_Allreduce(local_nPoints.data(), global_nPoints.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sumD.data(), global_sumD.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sumV.data(), global_sumV.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sumE.data(), global_sumE.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        // Update centroids and watch for convergence (ONLY ROOT)
        bool localConverged = true;
        std::vector<Point> newCentroids(k);

        if (world_rank == 0){
            for (int clusterId = 0; clusterId < k; ++clusterId)
            {
                if (global_nPoints[clusterId] == 0) {
                    newCentroids[clusterId] = centroids[clusterId]; //Catches unnamed 
                    
                    continue;
                }

                newCentroids[clusterId].danceability = global_sumD[clusterId] / global_nPoints[clusterId];
                newCentroids[clusterId].valence      = global_sumV[clusterId] / global_nPoints[clusterId];
                newCentroids[clusterId].energy       = global_sumE[clusterId] / global_nPoints[clusterId];
                
                double delta = centroids[clusterId].distance(newCentroids[clusterId]);
                if (delta > 1e-4){
                    localConverged = false;
                }
            }
            centroids = newCentroids;
        }

        // Broadcast updated centroids
        MPI_Bcast(newCentroids.data(), k * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

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
    // start timer
    const auto start = std::chrono::high_resolution_clock::now();

    // Set up MPI for Distribution stuff
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // first check for proper command line args
    if (argc != 4) {
        if (world_rank == 0)
            std::cerr << "Usage: " << argv[0] << " <input_file> <k> <thread_count>\n";
        MPI_Finalize();
        return 1;
    }
    
    // get thread num from args
    std::string inputFile = argv[1];
    int k = std::stoi(argv[2]);
    int threads = std::stoi(argv[3]);
    int epochs = 100; // number of iterations

    std::vector<Point> allPoints;
    if (world_rank == 0)
        allPoints = readcsv(inputFile);

    if (allPoints.empty())
    {
        std::cerr << "No data points loaded. Check your input file.\n";
        return 1;
    }
    
    // Broadcast size to all
    int totalSize = allPoints.size();
    MPI_Bcast(&totalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // set number of threads
    omp_set_num_threads(threads);

    // Split data evenly among processes
    int localSize = totalSize / world_size;
    std::vector<Point> localPoints(localSize);
    MPI_Scatter(allPoints.data(), localSize * sizeof(Point), MPI_BYTE,
                localPoints.data(), localSize * sizeof(Point), MPI_BYTE,
                0, MPI_COMM_WORLD);

    omp_set_num_threads(threads);
    kMeansClustering(localPoints, epochs, k, world_rank, world_size);

    // Gather all points at root
    MPI_Gather(localPoints.data(), localSize * sizeof(Point), MPI_BYTE,
               allPoints.data(), localSize * sizeof(Point), MPI_BYTE,
               0, MPI_COMM_WORLD);

    // Write results to output file and end and calculator timer for time
    if (world_rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::ofstream myfile("distributed_output.csv");
        myfile << "danceability,valence,energy,cluster\n";
        for (const auto &point : allPoints)
            myfile << point.danceability << "," << point.valence << "," << point.energy << "," << point.cluster << "\n";
        myfile.close();
        std::cout << "Results saved to distributed_output.csv\n";
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
