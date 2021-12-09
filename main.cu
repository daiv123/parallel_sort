
#include "helper.hpp"

#include "merge.hu"
#include "sort.hu"


namespace gpu_algorithms_labs_evaluation {

enum Mode { GPU_SORT_TILES, GPU_MERGE_TILES, GPU_SORT_BASIC, GPU_SORT_TILED};


void cpu_sort_tiles(std::vector<int> &data, int block_size) {
    int n = data.size();
    int num_tiles = n / block_size + (n % block_size == 0 ? 0 : 1);
    for (int i = 0; i < num_tiles; i++) {
        int start = i * block_size;
        int end = (i + 1) * block_size;
        end = end > n ? n : end;
        std::sort(data.begin() + start, data.begin() + end);
    }
}
void merge_sequential_host(int* A, int A_len, int* B, int B_len, int* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
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
void cpu_merge_chunks(int* A, int n, int chunk_size, int* C) {
    int num_chunks = n / chunk_size + (n % chunk_size == 0 ? 0 : 1);
    for (int i = 0; i < num_chunks; i+=2) {
        int start = i * chunk_size;
        int middle = (i + 1) * chunk_size;
        int end = (i + 2) * chunk_size;
        end = end > n ? n : end;
        merge_sequential_host(A + start, middle - start, A + middle, end - middle, C + start);
    }
}

void eval(int len, int chunk_size, Mode mode) {

        // Initialize variables
    // ----------------------------------------------
    const int data_byteCnt = len * sizeof(int);
    const int block_size = 64;

    std::vector<int> data_h(len);
    std::vector<int> solution(len);

    //GPU
    int* data_d; // Input array
    int* out_d; // Output array

    timer_start("Generating test data and reference solution");

    std::default_random_engine gen;

    std::uniform_int_distribution<int> dis(-50, 100);
    

    for (int i = 0; i < len; i++) {
        data_h[i] = dis(gen);
    }

    timer_start("Computing reference solution");
    if (mode == GPU_SORT_TILES) {
        solution = data_h;
        cpu_sort_tiles(solution, block_size);
    }
    else if (mode == GPU_MERGE_TILES) {
        cpu_sort_tiles(data_h, chunk_size);
        cpu_merge_chunks(data_h.data(), len, chunk_size, solution.data());
    }
    else if (mode == GPU_SORT_BASIC || mode == GPU_SORT_TILED) {
        solution = data_h;
        std::sort(solution.begin(), solution.end());
    }
    timer_stop();

    // for (int i = 0; i < len; i++) {
    //     printf("%d ", data_h[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < len; i++) {
    //     printf("%d ", solution[i]);
    // }
    // printf("\n");
    // printf("\n");

    timer_start("Allocating GPU memory");
    CUDA_RUNTIME(cudaMalloc((void**)&data_d, data_byteCnt));
    CUDA_RUNTIME(cudaMalloc((void**)&out_d, data_byteCnt));
    timer_stop();

    timer_start("Copying inputs to the GPU");
    CUDA_RUNTIME(cudaMemcpy(data_d, data_h.data(), data_byteCnt, cudaMemcpyHostToDevice));
    timer_stop();

    timer_start("Executing GPU kernel");
    if (mode == GPU_SORT_TILES) {
        gpu_sort_tiles(data_d, len);
    }
    else if (mode == GPU_MERGE_TILES) {
        gpu_tiled_merge_tiles(data_d, len, out_d, chunk_size);
    }
    else if (mode == GPU_SORT_BASIC) {
        gpu_sort_basic(data_d, len);
    }
    else if (mode == GPU_SORT_TILED) {
        gpu_sort_tiled(data_d, len);
    }
    
    timer_stop();

    timer_start("Copying output to host");
    if (mode == GPU_SORT_TILES) {
        CUDA_RUNTIME(cudaMemcpy(data_h.data(), data_d, data_byteCnt, cudaMemcpyDeviceToHost));
    }
    else if (mode == GPU_MERGE_TILES) {
        CUDA_RUNTIME(cudaMemcpy(data_h.data(), out_d, data_byteCnt, cudaMemcpyDeviceToHost));
    }
    else if (mode == GPU_SORT_BASIC || mode == GPU_SORT_TILED) {
        CUDA_RUNTIME(cudaMemcpy(data_h.data(), data_d, data_byteCnt, cudaMemcpyDeviceToHost));
    }
    
    timer_stop();

    // for(int i = 0; i < len; i++) {
    //     printf("%d ", data_h[i]);
    // }
    // printf("\n");

    timer_start("Verifying results");
    verify(solution, data_h);
    timer_stop();

    
    CUDA_RUNTIME(cudaFree(data_d));
}

TEST_CASE("required", "[GPU_MERGE]") {
    // SECTION("SORT_CHUNKS 50") { eval(50,0, GPU_SORT_TILES); }
    // SECTION("SORT_CHUNKS 10000") { eval(10000,0, GPU_SORT_TILES); }

    SECTION("MERGE_CHUNKS 16 4") { eval(16,4, GPU_MERGE_TILES); }
    SECTION("MERGE_CHUNKS 9 8") { eval(9,8, GPU_MERGE_TILES); }
    // SECTION("MERGE_CHUNKS 10000 20") { eval(10000,16, GPU_MERGE_TILES); }

    // SECTION("Sort 9") { eval(10,0, GPU_SORT_BASIC); }
    // SECTION("Sort 1048576") { eval(1048576,0, GPU_SORT_BASIC); }
    SECTION("Sort 4000000") { eval(4000000,0, GPU_SORT_BASIC); }

    SECTION("Sort 4000000") { eval(4000000,0, GPU_SORT_TILED); }


}


} // namespace gpu_algorithms_labs_evaluation