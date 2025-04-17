## 1. Instructions for how to build and execute on CHPC
- Start by running the Makefile which will create executables for all implementations.

## 2. Description of the approach used for each of the following implementations
1. Serial
2. Parallel shared memory CPU
3. Parallel CUDA GPU 
4. Distributed Memory CPU
5. Distributed memory GPU

## 3. Scaling study experiments where you compare implementations
- 1 vs 2 
- 3 (note: No scaling study for GPUs, instead look at different block size)
- 4 vs 5 (note: these will have to use from 2 to 4 nodes of any of the CHPC clusters)

## 4. Usea  validation function to check that the result from parallel implementations is equal to the serial output implementation

## 5. reuse code across implementations
utils.cpp and utils.hpp

## 6. Visualization of the output
Refer to the 3d_clusters.png

## 7. Clearly explain who was responsible for which task on the project 
Kade: Serial and Parallel shared memory implementations, as well as the python script for visualization. 
Rett: GPU implementation, as well as the util files and Makefile.


## References
[Serial Implementation Tutorial](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
