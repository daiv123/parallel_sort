#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "merge.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

#define comp(A,B) (A < B)

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
 template <typename T>
 __device__ void merge_sequential(T* A, int A_len, T* B, int B_len, T* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = comp(A[i],B[j]) ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

// co-rank
template <typename T>
__device__ int co_rank (int k, T* A, int m, T* B, int n) {

    int low = (k>n ? k-n : 0);
    int high = (k<m ? k : m);
    while (low < high) {
        int i = low + (high - low) / 2;
        int j = k - i;
        if (i > 0 && j < n && !comp(A[i-1],B[j])) {
            high = i -1;
        } else if (j > 0 && i < m && comp(A[i],B[j-1])) {
            low = i + 1;
        } else {
            return i;
        }
    }
    return low;
}
/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */

__device__ void gpu_merge_basic_kernel(int* A, int A_len, int* B, int B_len, int* C, int tid, int elt) {

    int k_curr = tid * elt;
    if (A_len + B_len < k_curr) { k_curr = A_len + B_len; }

    int k_next = k_curr + elt;
    if (A_len + B_len < k_next) { k_next = A_len + B_len; }

    int i_curr = co_rank (k_curr, A, A_len, B, B_len);
    int i_next = co_rank (k_next, A, A_len, B, B_len);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential (&A[i_curr], i_next - i_curr,
                      &B[j_curr], j_next - j_curr,
                      &C[k_curr]);
}
__global__ void gpu_merge_basic_chunked_kernel(int* A, int A_len, int* C, int chunk_size) {
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     for (int i = 0; i < A_len; i++) {
    //         printf("%d ", A[i]);
    //     }
    //     printf(" for chunksize %d\n", chunk_size);
    // }
    __syncthreads();
    int num_chunks = ceil_div(A_len, chunk_size);
    int num_sections = num_chunks / 2;
    int blocks_per_section = gridDim.x / num_sections;

    if (blockIdx.x < blocks_per_section * num_sections) {
        int section_id = blockIdx.x / blocks_per_section;

        int L_idx = section_id * chunk_size * 2;
        int L_len = chunk_size;
        int R_idx = L_idx + L_len;
        int R_len = chunk_size;
        if (section_id * chunk_size * 2 + chunk_size * 2 > A_len) {
            R_len = A_len - (section_id * chunk_size * 2) - chunk_size;
        }

        int O_idx = section_id * chunk_size * 2;
        // if (threadIdx.x == 0) {
        //     printf("%d %d %d %d %d %d %d %d \n", L_idx, L_len, R_idx, R_len, O_idx, section_id, blocks_per_section, num_sections);
        // }
        int tid = (blockIdx.x % blocks_per_section) * blockDim.x + threadIdx.x;
        int elt = ceil ((L_len + R_len)*1.0f/(blockDim.x*blocks_per_section));
        __syncthreads();
        gpu_merge_basic_kernel(A + L_idx, L_len, A + R_idx, R_len, C + O_idx, tid, elt);
    }
    __syncthreads();
    int remaining = A_len - (num_sections * chunk_size * 2);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = num_sections * chunk_size * 2;
    for (int i = tid; i < remaining; i += blockDim.x * gridDim.x) {
        C[i + start] = A[i + start];
    }


    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     for (int i = 0; i < A_len; i++) {
    //         printf("%d ", C[i]);
    //     }
    //     printf(" for chunksize %d\n", chunk_size);
    // }
    
}
/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */

__device__ void gpu_merge_tiled_kernel(int* A, int A_len, int* B, int B_len, int* C, int gridD, int blockI,
                                        int* shared_A, int* shared_B, int* shared_C, int* shared_A_idx) {
    /* Your code here */

    int elt = ceil ((A_len+B_len)*1.0f/(gridD));

    int c_blk_curr = blockI * elt;
    int c_blk_next = c_blk_curr + elt;
    if (A_len + B_len < c_blk_next) { c_blk_next = A_len + B_len; }

    if (threadIdx.x == 0) {
        shared_A_idx[0] = co_rank (c_blk_curr, A, A_len, B, B_len);
        shared_A_idx[1] = co_rank (c_blk_next, A, A_len, B, B_len);
    }
    __syncthreads();

    int a_blk_curr = shared_A_idx[0];
    int a_blk_next = shared_A_idx[1];
    int b_blk_curr = c_blk_curr - a_blk_curr;
    int b_blk_next = c_blk_next - a_blk_next;

    __syncthreads();

    int a_length = a_blk_next - a_blk_curr;
    int b_length = b_blk_next - b_blk_curr;
    int c_length = c_blk_next - c_blk_curr;

    int a_consumed = 0;
    int b_consumed = 0;
    int c_produced = 0;

    int num_tiles = ceil_div(c_length, TILE_SIZE);

    for (int tile_num = 0; tile_num < num_tiles; tile_num++) {
        
        for (int i = 0; i < TILE_SIZE; i+= blockDim.x) {
            if (i + threadIdx.x < a_length - a_consumed) {
                shared_A[i + threadIdx.x] = 
                    A[a_blk_curr + a_consumed + i + threadIdx.x];
            }
            if (i + threadIdx.x < b_length - b_consumed) {
                shared_B[i + threadIdx.x] = 
                    B[b_blk_curr + b_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int per_thread = TILE_SIZE / blockDim.x;
        int c_thr_curr = threadIdx.x * per_thread;
        int c_thr_next = c_thr_curr + per_thread;
        if (c_thr_next > c_length) { c_thr_next = c_length; }
        
        int c_remaining = c_length - c_produced;
        if (c_remaining < c_thr_curr) { c_thr_curr = c_remaining; }
        if (c_remaining < c_thr_next) { c_thr_next = c_remaining; }

        int a_in_tile = a_length - a_consumed;
        if (TILE_SIZE < a_in_tile) { a_in_tile = TILE_SIZE; }
        int b_in_tile = b_length - b_consumed;
        if (TILE_SIZE < b_in_tile) { b_in_tile = TILE_SIZE; }

        int a_thr_curr = co_rank (c_thr_curr, shared_A, a_in_tile, shared_B, b_in_tile);
        int a_thr_next = co_rank (c_thr_next, shared_A, a_in_tile, shared_B, b_in_tile);
        int b_thr_curr = c_thr_curr - a_thr_curr;
        int b_thr_next = c_thr_next - a_thr_next;

        merge_sequential(shared_A + a_thr_curr, a_thr_next - a_thr_curr,
                        shared_B + b_thr_curr, b_thr_next - b_thr_curr,
                        shared_C + c_thr_curr);

        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i+= blockDim.x) {
            if (i + threadIdx.x < c_length - c_produced) {
                C[c_blk_curr + c_produced + i + threadIdx.x] = 
                    shared_C[i + threadIdx.x];
            }
        }

        if (tile_num < num_tiles - 1) {
            int a_tile_consumed = co_rank(TILE_SIZE, shared_A, a_in_tile, shared_B, b_in_tile);
            a_consumed += a_tile_consumed;
            b_consumed += TILE_SIZE - a_tile_consumed;
            c_produced += TILE_SIZE;
        }
        __syncthreads();
    }

}
__global__ void gpu_merge_tiled_chunked_kernel(int* A, int A_len, int* C, int chunk_size) {
    
    __shared__ int shared_A[TILE_SIZE];
    __shared__ int shared_B[TILE_SIZE];
    __shared__ int shared_C[TILE_SIZE];
    __shared__ int shared_A_idx[2];
    
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     for (int i = 0; i < A_len; i++) {
    //         printf("%d ", A[i]);
    //     }
    //     printf(" for chunksize %d\n", chunk_size);
    // }
    // __syncthreads();
    int num_chunks = ceil_div(A_len, chunk_size);
    int num_sections = num_chunks / 2;
    int blocks_per_section = gridDim.x / num_sections;

    if (blockIdx.x < blocks_per_section * num_sections) {
        int section_id = blockIdx.x / blocks_per_section;

        int L_idx = section_id * chunk_size * 2;
        int L_len = chunk_size;
        int R_idx = L_idx + L_len;
        int R_len = chunk_size;
        if (section_id * chunk_size * 2 + chunk_size * 2 > A_len) {
            R_len = A_len - (section_id * chunk_size * 2) - chunk_size;
        }

        int O_idx = section_id * chunk_size * 2;
        // if (threadIdx.x == 0) {
        //     printf("%d %d %d %d %d %d %d %d \n", L_idx, L_len, R_idx, R_len, O_idx, section_id, blocks_per_section, num_sections);
        // }
        int gridD = blocks_per_section;
        int blockI = blockIdx.x % blocks_per_section;
        __syncthreads();
        gpu_merge_tiled_kernel(A + L_idx, L_len, A + R_idx, R_len, C + O_idx, gridD, blockI,
                                shared_A, shared_B, shared_C, shared_A_idx);
    }
    __syncthreads();
    int remaining = A_len - (num_sections * chunk_size * 2);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = num_sections * chunk_size * 2;
    for (int i = tid; i < remaining; i += blockDim.x * gridDim.x) {
        C[i + start] = A[i + start];
    }


    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     for (int i = 0; i < A_len; i++) {
    //         printf("%d ", C[i]);
    //     }
    //     printf(" for chunksize %d\n", chunk_size);
    // }
    
}
/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(int* A, int A_len, int* B, int B_len, int* C) {
    /* Your code here */
}


/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge_tiles(int* A, int A_len, int* C, int chunk_size) {
    const int blocks_per_section = chunk_size > TILE_SIZE ? chunk_size / (TILE_SIZE) : 1;
    const int num_sections = ceil_div(A_len, (chunk_size*2));
    const int numBlocks = blocks_per_section * num_sections;
    gpu_merge_basic_chunked_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, C, chunk_size);
}

void gpu_tiled_merge_tiles(int* A, int A_len, int* C, int chunk_size) {
    const int blocks_per_section = chunk_size > TILE_SIZE*4 ? chunk_size / (TILE_SIZE*4) : 1;
    const int num_sections = ceil_div(A_len, (chunk_size*2));
    const int numBlocks = blocks_per_section * num_sections;
    gpu_merge_tiled_chunked_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, C, chunk_size);
}

void gpu_circular_buffer_merge(int* A, int A_len, int* B, int B_len, int* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
