## 1. Instructions for how to build and execute on CHPC
- Start by running the Makefile which will create executables for all implementations.

## 2. Description of the approach used for each of the following implementations
1. Serial
2. Parallel shared memory CPU
3. Parallel CUDA GPU 
4. Distributed Memory CPU
    - This versioon distrbutes the k-means clustering algorithm across multiple process using MPI. Utilizing MPI_Scatter to evenly split the clusters among the processes available. The results are then aggregated globally using MPI_Allreduce, and centroids are updated and broadcasted only by the root process to avoid dangerous situations. And then the results are gathered using MPI_Gather also by the root process. This version maintains the core k-means logic while enabling scalability across multiple nodes in a distributed system.
5. Distributed memory GPU

## 3. Scaling study experiments where you compare implementations
- 1 vs 2 
- 3 (note: No scaling study for GPUs, instead look at different block size)
- 4 vs 5 (note: these will have to use from 2 to 4 nodes of any of the CHPC clusters)
| Number of Clusters (1064 threads) | Timing in Seconds CPU| Timing in Seconds GPU |
|----------|----------|----------|
| 2 | 36.7596 | 37.7818   |
| 4 | 43.2032 | 41.4922 |
| 8 | 113.039   | 103.287 |
| 16 | 391.539 | 354.859  |

- 
| Threads (per block for gpu, total in cpu)| Timing in Seconds CPU| Timing in Milliseconds GPU|
|----------|----------|----------|
| 4 | 35.5861 | 38.2805 |
| 8 | 36.0909 | 37.9196 |
| 16 | 36.4038 | 38.2282 |
| 32 | 46.2721 | 38.1518 |
| 64 | 83.7466 | 38.2738 |
| 128 | 36.6428  | 38.6429 |
| 256 | 36.2209 | 37.7776 |
| 512 | 36.4113 | 38.2375 |
| 1024 | 36.6035 | 37.8799 |

## 4. Usea  validation function to check that the result from parallel implementations is equal to the serial output implementation

## 5. reuse code across implementations
utils.cpp and utils.hpp

## 6. Visualization of the output
Refer to the 3d_clusters.png

## 7. Clearly explain who was responsible for which task on the project 
Kade: Serial and Parallel shared memory implementations, as well as the python script for visualization. \
Rett: GPU implementations, as well as the util files and Makefile.\
Rebecca: Distributed implementation.

## References
[Serial Implementation Tutorial](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
