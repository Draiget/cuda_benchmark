// ReSharper disable IdentifierTypo
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <ctime>

#define RAND_SEED_KERNEL 1504
#define CUDA_GRID_SIZE 256
#define CUDA_BLOCK_SIZE 64

struct mem_test_block
{
    size_t p_start;
    size_t p_end;
    size_t element_count;
    int* ptr;

    mem_test_block() {
        p_start = 0;
        p_end = 0;
        element_count = 0;
        ptr = nullptr;
    }

    mem_test_block(size_t start, size_t end, int* ptr) {
        this->p_start = start;
        this->p_end = end;
        this->element_count = floorf((end - start) / sizeof(int)) - 1;
        this->ptr = ptr;
    }
};

cudaError_t perform_memory_test(mem_test_block* mem_blocks, int mem_blocks_count, int iterations, float** took_time_stats, float* took_time_all, int tests_count);
cudaError_t perform_test_with_stats(mem_test_block* mem_blocks, int mem_blocks_count, int iterations, int tests_count);

__device__ int kernel_rand_generate(curandState* random_state, int ind, const int min, const size_t max) {
	auto local_state = random_state[ind];
    auto value = curand_uniform(&local_state);

    value *= (max - min + 0.999999);
    value += min;

    return static_cast<int>(truncf(value));
}

__global__ void mem_kernel_setup(curandState* state) 
{
	const int id = threadIdx.x;
    curand_init(RAND_SEED_KERNEL, id, 0, &state[id]);
}

__global__ void mem_kernel_process(curandState* random_state, mem_test_block* mem_test_blocks, const int mem_blocks_count, int iterations)
{
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const auto block_index = idx % mem_blocks_count;
    if (block_index >= mem_blocks_count) {
        printf("mem_kernel_process: Memory region by index %d out of bounds\n", block_index);
        return;
    }

	const auto mem_block = mem_test_blocks[block_index];
    //printf("mem_kernel_process: mem_block[%d].element_count = %lld\n", block_index, mem_block.element_count);

    int a = 0;
    int reads = 0;
    if (mem_blocks_count == 1) {
        reads = 2;
    }
    else {
        reads = 1;
    }

    for (auto i = 0; i < iterations; ++i) {
        // Get random memory index within our block
        const auto random_mem_index = kernel_rand_generate(random_state, idx, 0, mem_block.element_count);
        if (random_mem_index < 0) {
            printf("mem_kernel_process: Trying to access invalid memory block index = %d\n", random_mem_index);
            return;
        }

        //printf("mem_kernel_process: mem_block[%d] at index %d\n", random_mem_index, block_index);
        // Write access to memory within our block

        for (auto k = 0; k < reads; ++k) {
            a = mem_block.ptr[random_mem_index]; // = kernel_rand_generate(random_state, idx, 0, 1024);
        }
	}

    a *= 2;
}

int main()
{
    size_t freeMem, totalMem;
    auto cuda_status = cudaMemGetInfo(&freeMem, &totalMem);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Failed to get GPU memory information! (cudaMemGetInfo = %d)\n", cuda_status);
        return 1;
    }

    const auto test_iterations = 30000;
    const auto test_count = 50;

    const auto used_memory = totalMem - freeMem;
    const auto cuda_memory_block_size = freeMem - (used_memory * 2);
    printf("CUDA Memory [free=%llu Mb, total=%llu Mb, used=%llu Mb, allocated=%llu]\n", freeMem / 1024 / 1024, totalMem / 1024 / 1024, used_memory / 1024 / 1024, cuda_memory_block_size / 1024 / 1024);

    // Allocate all free memory to occupy bigger block
    int* cuda_memory_block = nullptr;
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&cuda_memory_block), cuda_memory_block_size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Allocation failed for CUDA block! (cudaMalloc = %d)\n", cuda_status);
        return 1;
    }

    const auto test_1_mem_blocks = static_cast<mem_test_block*>(malloc(sizeof(mem_test_block) * 1));
    test_1_mem_blocks[0] = mem_test_block{ 0, (size_t)(1024 * 1024 * 1024 * 11) - 1024, cuda_memory_block };
    
    cuda_status = perform_test_with_stats(test_1_mem_blocks, 1, test_iterations, test_count);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "#1 memory test failed!\n");
        cudaFree(cuda_memory_block);
        return 1;
    }
    
    free(test_1_mem_blocks);

    const auto test_2_mem_blocks = static_cast<mem_test_block*>(malloc(sizeof(mem_test_block) * 2));
    test_2_mem_blocks[0] = mem_test_block{ 0, (size_t)(1024 * 1024 * 1024 * 11) - 1024, cuda_memory_block };
    test_2_mem_blocks[1] = mem_test_block{ cuda_memory_block_size - (size_t)((1024 * 1024 * 1024 * 11) - 1024), cuda_memory_block_size, cuda_memory_block };

    cuda_status = perform_test_with_stats(test_2_mem_blocks, 2, test_iterations, test_count);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "#2 memory test failed!\n");
        cudaFree(cuda_memory_block);
        return 1;
    }

    free(test_2_mem_blocks);

    const auto test_3_mem_blocks = static_cast<mem_test_block*>(malloc(sizeof(mem_test_block) * 1));
    test_3_mem_blocks[0] = mem_test_block{ cuda_memory_block_size - (size_t)((1024 * 1024 * 1024 * 11) - 1024), cuda_memory_block_size, cuda_memory_block };
    
    cuda_status = perform_test_with_stats(test_3_mem_blocks, 1, test_iterations, test_count);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "#3 memory test failed!\n");
        cudaFree(cuda_memory_block);
        return 1;
    }
    
    free(test_3_mem_blocks);
    
    //const auto test_4_mem_blocks = static_cast<mem_test_block*>(malloc(sizeof(mem_test_block) * 1));
    //test_4_mem_blocks[0] = mem_test_block{ 0, (1024 * 1024 * 1024 * 2) - 1024, cuda_memory_block };
    //
    //cuda_status = perform_test_with_stats(test_4_mem_blocks, 1, test_iterations, test_count);
    //if (cuda_status != cudaSuccess) {
    //    fprintf(stderr, "#4 memory test failed!\n");
    //    cudaFree(cuda_memory_block);
    //    return 1;
    //}
    //
    //free(test_4_mem_blocks);

    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        cudaFree(cuda_memory_block);
        return 1;
    }

    cudaFree(cuda_memory_block);
    return 0;
}

cudaError_t perform_test_with_stats(mem_test_block* mem_blocks, int mem_blocks_count, const int iterations, const int tests_count) {
    float time_took_tests_all = 0;
    float* time_took_tests;

    const auto cuda_status = perform_memory_test(mem_blocks, mem_blocks_count, iterations, &time_took_tests, &time_took_tests_all, tests_count);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "** FAILED to perform memory test with %d regions of a memory block.\n", mem_blocks_count);
        return cuda_status;
    }

    float tests_took_total = 0;
    for (auto i = 0; i < tests_count; ++i) {
        tests_took_total += time_took_tests[i];
    }

    printf("%d tests with %d iterations took - %f ms avg, %f ms total.\n\n", tests_count, iterations, tests_took_total / tests_count, time_took_tests_all);
    return cuda_status;
}

cudaError_t perform_memory_test(mem_test_block* mem_blocks, int mem_blocks_count, const int iterations, float** took_time_stats, float* took_time_all, int tests_count)
{
	auto cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed, no CUDA GPU? Lel.");
        return cuda_status;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Setting up memory kernel (memory_regions = %d) ...\n", mem_blocks_count);

    curandState* dev_states;
    cudaMalloc(&dev_states, iterations * sizeof(curandState));
    srand(time(nullptr));

    // Setup memory kernel
    mem_kernel_setup <<< CUDA_GRID_SIZE, CUDA_BLOCK_SIZE, 0 >>> (dev_states);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "mem_kernel_setup launch failed: %s\n", cudaGetErrorString(cuda_status));
        return cuda_status;
    }

    printf("Launching memory kernel (memory_regions = %d) ...\n", mem_blocks_count);

    *took_time_stats = static_cast<float*>(malloc(sizeof(float) * tests_count));
	const auto time_stat = *took_time_stats;
    for (auto i = 0; i < tests_count; ++i) {
        cudaEventRecord(start);

        // Copy shared memory to GPU
        mem_test_block* mem_blocks_cuda_ptr = nullptr;
        cuda_status = cudaMalloc(reinterpret_cast<void**>(&mem_blocks_cuda_ptr), sizeof(mem_test_block) * mem_blocks_count);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Allocation failed for memory access region (cudaMalloc = %d)\n", cuda_status);
            return cuda_status;
        }

        cuda_status = cudaMemcpy(mem_blocks_cuda_ptr, mem_blocks, sizeof(mem_test_block) * mem_blocks_count, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to copy shared memory to CUDA (cudaMemcpy = %d)\n", cuda_status);
            return cuda_status;
        }

        // Launch memory kernel
        mem_kernel_process <<< CUDA_GRID_SIZE, CUDA_BLOCK_SIZE, 0 >>> (dev_states, mem_blocks_cuda_ptr, mem_blocks_count, iterations);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "mem_kernel_process launch failed: %s\n", cudaGetErrorString(cuda_status));
            return cuda_status;
        }

        cudaEventElapsedTime(&time_stat[i], start, stop);
        printf("- test #%d took %f ms\n", i, time_stat[i]);
        *took_time_all += time_stat[i];
    }

    return cuda_status;
}
