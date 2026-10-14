/*
 * bench_scalar_mul.cu - throughput of k * G on the GPU.
 *
 * Every thread derives the public key for its own scalar. Usage:
 *   bench_scalar_mul [keys] [threads_per_block]
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_secp256k1/secp256k1.cuh>

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err__));          \
            exit(2);                                                                 \
        }                                                                            \
    } while (0)

__constant__ ec_group_t c_group;

/* Distinct, well-mixed scalars without any host-side input. */
__device__ bn_t scalar_for(int idx)
{
    bn_t k;
    bn_zero(&k);
    uint64_t x = 0x9E3779B97F4A7C15ULL * (uint64_t)(idx + 1);
    for (int i = 0; i < 4; i++) {
        x ^= x >> 31;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        k.d[i] = x;
    }
    k.top = 4;
    k.d[3] &= 0x7FFFFFFFFFFFFFFFULL; /* keep below the group order */
    return k;
}

__global__ void bench_ladder(uint64_t *sink, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    bn_t k = scalar_for(i);
    ec_affine_t pub;
    ec_point_mul(&c_group, &pub, &k, &c_group.g);
    sink[i] = pub.x.d[0];
}

__global__ void bench_vartime(uint64_t *sink, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    bn_t k = scalar_for(i);
    ec_point_t r;
    ec_affine_t pub;
    ec_point_mul_vartime(&c_group, &r, &k, &c_group.g);
    ec_point_to_affine(&c_group, &pub, &r);
    sink[i] = pub.x.d[0];
}

template <typename Kernel>
static void run(const char *name, Kernel kernel, uint64_t *sink, int n, int block)
{
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(sink, n); /* warm-up */
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(sink, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("%-28s %8d keys in %8.1f ms  = %10.0f keys/s\n", name, n, ms, n / (ms / 1000.0));
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 1 << 18;
    int block = argc > 2 ? atoi(argv[2]) : 128;

    ec_group_t group;
    ec_group_init_secp256k1(&group);
    CUDA_CHECK(cudaMemcpyToSymbol(c_group, &group, sizeof(group)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d SMs\n", prop.name, prop.multiProcessorCount);

    uint64_t *sink;
    CUDA_CHECK(cudaMalloc(&sink, n * sizeof(uint64_t)));
    run("ladder k*G (ec_point_mul)", bench_ladder, sink, n, block);
    run("double-and-add k*G (vartime)", bench_vartime, sink, n, block);
    CUDA_CHECK(cudaFree(sink));
    return 0;
}
