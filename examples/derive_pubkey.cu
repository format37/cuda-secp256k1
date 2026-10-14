/*
 * derive_pubkey.cu - compressed secp256k1 public keys for a batch of private keys.
 *
 * Usage:
 *   derive_pubkey <hex private key> ...
 *   derive_pubkey < keys.txt          (one hex private key per line)
 *
 * Prints "<private key> <compressed public key>" per input. For example,
 * private key 1 gives the generator, 0279be667e...16f81798.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <cuda_secp256k1/secp256k1.cuh>

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err__));          \
            exit(2);                                                                 \
        }                                                                            \
    } while (0)

struct Result {
    uint8_t pub[33];
    int ok;
};

__constant__ ec_group_t c_group;

__global__ void derive(const bn_t *priv, Result *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    ec_affine_t pub;
    out[i].ok = ec_pubkey_create(&c_group, &pub, &priv[i]);
    if (out[i].ok)
        ec_affine_to_bytes(&pub, out[i].pub, 1);
}

int main(int argc, char **argv)
{
    std::vector<std::string> lines;
    for (int i = 1; i < argc; i++)
        lines.push_back(argv[i]);
    if (lines.empty()) {
        char buf[256];
        while (fgets(buf, sizeof(buf), stdin)) {
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
                s.pop_back();
            if (!s.empty())
                lines.push_back(s);
        }
    }
    if (lines.empty()) {
        fprintf(stderr, "usage: derive_pubkey <hex private key> ...\n");
        return 1;
    }

    std::vector<bn_t> priv(lines.size());
    for (size_t i = 0; i < lines.size(); i++) {
        if (!bn_from_hex(&priv[i], lines[i].c_str())) {
            fprintf(stderr, "not a hex number: %s\n", lines[i].c_str());
            return 1;
        }
    }

    ec_group_t group;
    ec_group_init_secp256k1(&group);
    CUDA_CHECK(cudaMemcpyToSymbol(c_group, &group, sizeof(group)));

    int n = (int)priv.size();
    bn_t *d_priv;
    Result *d_out;
    CUDA_CHECK(cudaMalloc(&d_priv, n * sizeof(bn_t)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Result)));
    CUDA_CHECK(cudaMemcpy(d_priv, priv.data(), n * sizeof(bn_t), cudaMemcpyHostToDevice));
    derive<<<(n + 127) / 128, 128>>>(d_priv, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Result> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(Result), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_priv));
    CUDA_CHECK(cudaFree(d_out));

    for (int i = 0; i < n; i++) {
        printf("%s ", lines[i].c_str());
        if (!out[i].ok) {
            printf("invalid private key (must be in [1, n-1])\n");
            continue;
        }
        for (int j = 0; j < 33; j++)
            printf("%02x", out[i].pub[j]);
        printf("\n");
    }
    return 0;
}
