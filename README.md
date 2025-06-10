# Accelerated-Pairwise-Mutual-Information-Calculation
The primary goal of this script is to calculate the pairwise mutual information for every possible combination of columns in a given dataset, transforming a process that could take hours into one that completes in minutes or seconds.


1. Mutual information measures the dependency between two variables; a higher value indicates a stronger relationship. This is a common task in feature selection and exploratory data analysis. The script's key feature is its performance. By using a custom Numba CUDA kernel, it parallelizes the entire computation on the GPU.


2. Dependencies
The script relies on the RAPIDS AI ecosystem and Numba. Ensure the following libraries are installed in your CUDA-enabled environment:

cudf: A Pandas-like GPU DataFrame library.

cupy: A NumPy-like GPU array library.

numba: A just-in-time compiler that translates Python functions to optimized machine code, including CUDA kernels.

tqdm: A utility for displaying progress bars.

3. Workflow Breakdown
The script operates in four main stages:

Step 1: Data Loading & Discretization
Mutual information is typically calculated on discrete variables. This stage prepares the data accordingly.

Load Data: The script loads a CSV file into a cudf DataFrame, which places the data directly onto the GPU's memory.

Discretize Continuous Variables: Since most real-world data is continuous, each column is converted into a set of discrete bins.

It calculates n_bins (e.g., 10) quantiles for each column.

It then uses digitize to map each value in the column to its corresponding bin index (0 to 9).

This results in a new DataFrame, df_discrete, where all values are integers representing bins.

Convert to CuPy Array: The discretized DataFrame is converted into a single, C-contiguous cupy array (data_array). This format is ideal for passing into the Numba CUDA kernel.

# Convert the entire dataframe to a single CuPy array for the kernel
data_array = cp.ascontiguousarray(df_discrete.values)
n_samples, n_features = data_array.shape

Step 2: The Numba CUDA Kernel (pairwise_mi_kernel)
This is the core of the acceleration. The @cuda.jit decorator compiles this Python function into a high-performance GPU kernel.

Grid-Stride Loop: The kernel is launched on a 2D grid of threads, where the coordinates (i, j) correspond to a pair of columns in the dataset. Each GPU thread is responsible for computing the MI for a single pair (column_i, column_j).

i, j = cuda.grid(2)
# Exit if the thread is outside the upper triangle of the matrix
if i >= n_features or j <= i:
    return

Local Contingency Matrix: For each thread, a small 10x10 contingency matrix is created in fast cuda.local.array. This matrix stores the joint frequency distribution (counts) of bin pairs for the two columns being analyzed. Using local memory is significantly faster than repeatedly accessing global GPU memory.

MI Calculation: Within each thread, the kernel performs the standard mutual information calculation based on the contingency matrix:

Calculates the joint probability p(x, y).

Calculates the marginal probabilities p(x) and p(y).

Computes the MI score using the formula: sum(p(x,y) * log( p(x,y) / (p(x) * p(y)) )).

Symmetric Write: The final MI score is written to both mi_matrix[i, j] and mi_matrix[j, i], filling the final matrix symmetrically.

Step 3: Kernel Launch
This section configures and launches the pairwise_mi_kernel.

Result Matrix: A cupy array (mi_matrix_gpu) is initialized with zeros to store the results.

Grid Configuration: The script calculates the required number of thread blocks to cover all column pairs. It defines a 2D block of 16x16=256 threads and then calculates how many blocks are needed in the x and y dimensions.

Kernel Execution: The kernel is launched with the configured grid and block dimensions, passing the input data and the output matrix as arguments.

Synchronization: cp.cuda.runtime.deviceSynchronize() is called to ensure the CPU waits for the GPU to finish all its calculations before proceeding.

# Launch the kernel!
pairwise_mi_kernel[blocks_per_grid, threads_per_block](data_array, mi_matrix_gpu)
cp.cuda.runtime.deviceSynchronize() # Wait for GPU to finish

Step 4: Post-processing and Output
Once the GPU computation is complete:

The MI matrix is moved from the cupy array back into a labeled cudf DataFrame.

The diagonal of the matrix is filled with 1.0, as the mutual information of a variable with itself is maximal.

The final DataFrame is converted to Pandas and saved to a CSV file.

4. How to Run the Script
Setup: Make sure you have a working RAPIDS environment with all the necessary dependencies installed.

Data: Place your merged_quarterly_data.csv file in the same directory as the script, or update the file path in the script.

Execute: Run the script from your terminal:

python pairwise_mi_accelerated.py

Output: The script will print its progress and save the final results to pairwise_mi_matrix_accelerated.csv.

5. Performance Gains Explained
The original script was slow because its nested Python for loops launched a separate GPU computation for each pair of columns. For 2791 columns, this means launching ~3.9 million individual cuml.metrics.mutual_info_score calls. The overhead of launching millions of kernels far outweighs the computation time of each one.

The Numba version achieves its speedup by:

Single Kernel Launch: It launches one single, massive kernel that covers all pairs. The GPU's hardware scheduler assigns the work across its thousands of cores, achieving true parallelism.

Eliminating CPU-GPU Overhead: The entire calculation is self-contained on the GPU, eliminating the back-and-forth communication between the CPU and GPU that plagued the original loop.

Optimized Memory Access: Using cuda.local.array leverages the fastest memory available on the GPU, minimizing memory latency within each thread's computation.
