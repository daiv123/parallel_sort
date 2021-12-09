#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "sort.hu"
#include "merge.hu"

#define BLOCK_SIZE 16
#define TILE_SIZE 64


__device__ inline bool comp(int A, int B) {
  return A < B;
}
// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}

//  merge, and merge_sort from https://www.geeksforgeeks.org/iterative-merge-sort/

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
__device__ void merge(int *arr, int l, int m, int r, int *buffer)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    int* L = arr + l;
    int* R = arr + m + 1;
 
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (comp(L[i], R[j]))
        {
            buffer[k] = L[i];
            i++;
        }
        else
        {
            buffer[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        buffer[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        buffer[k] = R[j];
        j++;
        k++;
    }
}

 
/* Iterative mergesort function to sort arr[0...n-1] */
__device__ void merge_sort(int *arr, int n, int *buffer)
{
    int curr_size;  // For current size of subarrays to be merged
                    // curr_size varies from 1 to n/2
    int left_start; // For picking starting index of left subarray
                    // to be merged
    int* original = arr;
    int* temp;

    // Merge subarrays in bottom up manner.  First merge subarrays of
    // size 1 to create sorted subarrays of size 2, then merge subarrays
    // of size 2 to create sorted subarrays of size 4, and so on.
    for (curr_size=1; curr_size<=n-1; curr_size = 2*curr_size)
    {
        // Pick starting point of different subarrays of current size
        for (left_start=0; left_start<n-1; left_start += 2*curr_size)
        {
            // Find ending point of left subarray. mid+1 is starting
            // point of right
            int mid = min(left_start + curr_size - 1, n-1);
    
            int right_end = min(left_start + 2*curr_size - 1, n-1);
    
            // Merge Subarrays arr[left_start...mid] & arr[mid+1...right_end]
            merge(arr, left_start, mid, right_end, buffer);
        }
        temp = buffer;
        buffer = arr;
        arr = temp;
    }
    if (original != arr) {
        for (int i = 0; i < n; i++) {
            original[i] = arr[i];
        }
    }
    
}

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_sort_tiles_kernel(int* A, int A_len) {
    __shared__ int shared_A[TILE_SIZE * BLOCK_SIZE];
    __shared__ int buffer[TILE_SIZE * BLOCK_SIZE];
    int b_idx = blockIdx.x * TILE_SIZE * BLOCK_SIZE;
    for (int i = threadIdx.x; i < TILE_SIZE * BLOCK_SIZE; i += blockDim.x) {
        if (i + b_idx < A_len) {
            shared_A[i] = A[i + b_idx];
        }
    }
    __syncthreads();
    int t_idx = threadIdx.x * TILE_SIZE;
    int t_len = b_idx + t_idx + TILE_SIZE < A_len ? TILE_SIZE : A_len - t_idx - b_idx;
    if (b_idx + t_idx < A_len) {
        merge_sort(shared_A + t_idx, t_len, buffer + t_idx);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TILE_SIZE * BLOCK_SIZE; i += blockDim.x) {
        if (i + b_idx < A_len) {
            A[i + b_idx] = shared_A[i];
        }
    }
}


/******************************************************************************
 Functions
*******************************************************************************/

void gpu_sort_tiles(int* A, int A_len) {
    const int numBlocks = ceil_div(A_len, TILE_SIZE*BLOCK_SIZE);
    gpu_sort_tiles_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len);
}

void gpu_sort_basic(int* A, int A_len) {
    gpu_sort_tiles(A, A_len);

    cudaDeviceSynchronize();

    int* original = A;
    int* buffer;
    cudaMalloc((void**)&buffer, A_len * sizeof(int));
    int* temp;

    int curr_size;  
    
    for (curr_size=TILE_SIZE; curr_size < A_len; curr_size = 2*curr_size)
    {
        gpu_basic_merge_tiles(A, A_len, buffer, curr_size);

        temp = buffer;
        buffer = A;
        A = temp;
    }
    if (original != A) {
        cudaMemcpy(original, A, A_len * sizeof(int), cudaMemcpyDeviceToDevice);
        buffer = A;
    }
    cudaFree(buffer);
}

void gpu_sort_tiled(int* A, int A_len) {
    gpu_sort_tiles(A, A_len);

    cudaDeviceSynchronize();

    int* original = A;
    int* buffer;
    cudaMalloc((void**)&buffer, A_len * sizeof(int));
    int* temp;

    int curr_size;  
    
    for (curr_size=TILE_SIZE; curr_size < A_len; curr_size = 2*curr_size)
    {
        gpu_tiled_merge_tiles(A, A_len, buffer, curr_size);

        temp = buffer;
        buffer = A;
        A = temp;
    }
    if (original != A) {
        cudaMemcpy(original, A, A_len * sizeof(int), cudaMemcpyDeviceToDevice);
        buffer = A;
    }
    cudaFree(buffer);
}