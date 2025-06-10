import cudf
import cupy as cp
from numba import cuda
import math
from tqdm import tqdm
import time

# --- 1. Load and Discretize Data  ---
print("Loading and discretizing data...")
n_bins = 10
# For demonstration, this is my dataframe. Use your file path.
allsec_gpu = cudf.read_csv('merged_quarterly_data.csv', index_col='DATE', parse_dates=['DATE'])


df_discrete = cudf.DataFrame()
for col in tqdm(allsec_gpu.columns, desc="Discretizing columns"):
    bins = allsec_gpu[col].quantile(cp.linspace(0, 1, n_bins + 1)).values
    df_discrete[col] = allsec_gpu[col].digitize(bins, right=True).astype('int32')

# Convert the entire dataframe to a single CuPy array for the kernel
# Using .values is fine, explicitly ensuring C-contiguous layout is best practice.
data_array = cp.ascontiguousarray(df_discrete.values)
n_samples, n_features = data_array.shape

# --- 2. Define the Numba CUDA Kernel for Pairwise MI ---
# This kernel will be compiled for the GPU.
# NOTE: For Numba to create optimized local arrays, N_BINS must be a constant.
N_BINS = 10

@cuda.jit
def pairwise_mi_kernel(data, mi_matrix):
    """
    Calculates the Mutual Information for all pairs of columns on the GPU.
    Each thread computes the MI for one (i, j) pair.
    """
    i, j = cuda.grid(2)

    # Exit if the thread is outside the upper triangle of the matrix
    if i >= n_features or j <= i:
        return

    # Use a small, fast local array for the contingency matrix of this thread's pair
    contingency_matrix = cuda.local.array((N_BINS, N_BINS), dtype=cp.float32)
    for x in range(N_BINS):
        for y in range(N_BINS):
            contingency_matrix[x, y] = 0.0

    # Calculate the contingency matrix (joint counts)
    for row_idx in range(n_samples):
        bin1 = data[row_idx, i]
        bin2 = data[row_idx, j]
        # Ensure bins are within the expected range before indexing
        if 0 <= bin1 < N_BINS and 0 <= bin2 < N_BINS:
            contingency_matrix[bin1, bin2] += 1

    # Calculate marginal probabilities (px and py)
    px = cuda.local.array(N_BINS, dtype=cp.float32)
    py = cuda.local.array(N_BINS, dtype=cp.float32)
    for x in range(N_BINS):
        px[x] = 0.0
        py[x] = 0.0

    for x in range(N_BINS):
        for y in range(N_BINS):
            px[x] += contingency_matrix[x, y]
            py[y] += contingency_matrix[x, y]

    # Calculate Mutual Information score
    mi_score = 0.0
    for x in range(N_BINS):
        for y in range(N_BINS):
            if contingency_matrix[x, y] > 0: # Avoid log(0)
                # MI formula: p(x,y) * log( p(x,y) / (p(x) * p(y)) )
                p_xy = contingency_matrix[x, y] / n_samples
                p_x = px[x] / n_samples
                p_y = py[y] / n_samples
                mi_score += p_xy * math.log(p_xy / (p_x * p_y))

    # Store the result in the final matrix (symmetric)
    mi_matrix[i, j] = mi_score
    mi_matrix[j, i] = mi_score


# --- 3. Launch the Kernel ---
print("Computing MI matrix on GPU with Numba...")
# Initialize the result matrix on the GPU
mi_matrix_gpu = cp.zeros((n_features, n_features), dtype=cp.float32)

# Configure the GPU grid dimensions
threads_per_block = (16, 16) # 256 threads per block
blocks_per_grid_x = math.ceil(n_features / threads_per_block[0])
blocks_per_grid_y = math.ceil(n_features / threads_per_block[1])
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the kernel!
start_time = time.time()
pairwise_mi_kernel[blocks_per_grid, threads_per_block](data_array, mi_matrix_gpu)
cp.cuda.runtime.deviceSynchronize() # Wait for GPU to finish
end_time = time.time()

print(f"MI computation finished in: {end_time - start_time:.2f} seconds")

# --- 4. Format and Save Results ---
# Set diagonal to 1.0 for clarity, as MI(X,X) is max entropy
cp.fill_diagonal(mi_matrix_gpu, 1.0)

# Convert to labeled DataFrame
mi_df = cudf.DataFrame(mi_matrix_gpu, index=df_discrete.columns, columns=df_discrete.columns)

# Save results
print("Saving results to CSV...")
mi_df.to_pandas().to_csv('pairwise_mi_matrix_accelerated.csv')
print("Done.")
