
# MLPerf HPC NVIDIA-Optimized Implementations

This is a repository of NVIDIA-optimized implementations for the MLPerf HPC benchmarks.

# Contents

These are the MLPerf HPC benchmarks:
* DeepCam
* Cosmoflow
* OC20

# Reference Implementations
The reference implementation are available at https://github.com/mlcommons/hpc 

Each reference implementation provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container.
* A script which downloads the appropriate dataset.
* A script which runs and times training the model.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

Follow the README under each benchmark to run each benchmark which will run until the target quality is reached and then stop, printing timing results. 
