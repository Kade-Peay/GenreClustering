# Genre Clustering of spotify data:
Implement a parallel K-Mean clustering algorithm
 Use Spotify data metrics for 1.2M> songs 
 https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs

## Genre Reveal Party: challenges
* Shared memory implementation (simple)
    * Find parts that can be parallelized with threads
* Distributed memory implementation
    * Domain decomposition (split data among cores)
    * Share centroids and update until convergence
    * Generate output and visualization
* GPU implementation
    * In principle similar to shared memory
