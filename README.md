## 1. Instructions for how to build and execute on CHPC
- Start by running the Makefile which will create executables for all implementations.
- Module load CUDA for the gpu implementation
- Usage
    * Serial: ./serial <input_file> <number_of_clusters>
    * Shared: ./shared <input_file> <number_of_clusters> <thread_count>
    * Shared-GPU: ./shared-gpu <input_file> <number_of_clusters> <threads_per_block>
    * Distributed CPU: ./distributed <input_file> <k> <thread_count> 
    * Distributed-GPU: ./distributed-gpu <input_file> <number_of_clusters> <threads_per_block>

## 2. Description of the approach used for each of the following implementations
1. Serial
    - This is the standard, single-threaded version of the k-means clustering that processes sequentially. This is the baseline C++ implementation using vectors to assign poitns to the clusters and update the centroids over the fixed 100 epochs.
2. Parallel shared memory CPU
    - This version parallelizes the k-means clustering algorithm using OpenMP to distribute the work across CPU cores. It has two parallel regions: Cluster aassignment via a nearest-neighbor search and the statistical reduction for centroid updates. It maintains thread safety using OpenMP's reduction operations while keeping the same base algorithm from the serial implementation. 
3. Parallel CUDA GPU 
4. Distributed Memory CPU
5. Distributed memory GPU

## 3. Scaling study experiments where you compare implementations
- 1 vs 2 
- 3 (note: No scaling study for GPUs, instead look at different block size)
- 4 vs 5 (note: these will have to use from 2 to 4 nodes of any of the CHPC clusters)

## 4. Use a  validation function to check that the result from parallel implementations is equal to the serial output implementation
- Serial vs. Shared: The files are identical
- Serial vs. GPU:
- Serial vs. Distributed: The files are different by 7.68%
- Serial vs. Distributed-GPU:

## 5. reuse code across implementations
utils.cpp and utils.hpp

## 6. Visualization of the output
This is done using the plotter.py script and outputs to a file named 3d_clusters.png.
![Cluster vis](3d_clusters.png)

## 7. Clearly explain who was responsible for which task on the project 
Kade: Serial and Parallel shared memory implementations, as well as the python script for visualization. 
Rett: GPU implementations, as well as the util files and Makefile.
Rebecca: Distributed implementation.

## References
[Serial Implementation Tutorial](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
