/*
 * mnemonic_to_address.cu - BIP39 mnemonic to a wallet address, one GPU thread per mnemonic.
 *
 * Usage:
 *   mnemonic_to_address [-f p2pkh|p2wpkh|avax] [-p passphrase] [-d path] "<mnemonic>" ...
 *   mnemonic_to_address [options] < mnemonics.txt                       (one per line)
 *
 * Formats and their default paths:
 *   p2pkh   Bitcoin legacy "1..."        m/44'/0'/0'/0/0
 *   p2wpkh  Bitcoin segwit "bc1q..."     m/84'/0'/0'/0/0
 *   avax    Avalanche P-chain "P-avax1..." m/44'/9000'/0'/0/0
 *
 * Prints "<address> <mnemonic>" per input. Mnemonics are used as given:
 * there is no word-list or checksum validation.
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

#define MAX_MNEMONIC 256
#define MAX_DEPTH 16

enum Format { P2PKH, P2WPKH, AVAX };

struct Mnemonic {
    char text[MAX_MNEMONIC];
    int len;
};

struct Result {
    char address[BECH32_MAX_LEN + 1];
    int ok;
};

__constant__ ec_group_t c_group;
__constant__ char c_passphrase[BIP39_MAX_PASSPHRASE];
__constant__ int c_passphrase_len;
__constant__ uint32_t c_path[MAX_DEPTH];
__constant__ int c_depth;
__constant__ int c_format;

__global__ void derive(const Mnemonic *in, Result *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint8_t seed[BIP39_SEED_SIZE], pub[33];
    bip32_key_t k;
    Result r;
    r.ok = bip39_seed(in[i].text, in[i].len, c_passphrase, c_passphrase_len, seed)
           && bip32_master(&c_group, &k, seed, sizeof(seed)) && bip32_derive_path(&c_group, &k, c_path, c_depth)
           && bip32_pubkey(&c_group, &k, pub);
    if (r.ok) {
        if (c_format == P2PKH)
            r.ok = address_p2pkh(0x00, pub, r.address, sizeof(r.address)) > 0;
        else if (c_format == P2WPKH)
            r.ok = address_p2wpkh("bc", pub, r.address, sizeof(r.address)) > 0;
        else
            r.ok = address_avalanche("avax", pub, r.address, sizeof(r.address)) > 0;
    }
    out[i] = r;
}

int main(int argc, char **argv)
{
    const char *passphrase = "";
    const char *path = NULL;
    const char *format_name = "p2pkh";
    std::vector<std::string> lines;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc)
            passphrase = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc)
            path = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc)
            format_name = argv[++i];
        else
            lines.push_back(argv[i]);
    }
    int format;
    const char *prefix = "";
    if (!strcmp(format_name, "p2pkh")) {
        format = P2PKH;
        if (!path)
            path = "m/44'/0'/0'/0/0";
    } else if (!strcmp(format_name, "p2wpkh")) {
        format = P2WPKH;
        if (!path)
            path = "m/84'/0'/0'/0/0";
    } else if (!strcmp(format_name, "avax")) {
        format = AVAX;
        prefix = "P-";
        if (!path)
            path = "m/44'/9000'/0'/0/0";
    } else {
        fprintf(stderr, "unknown format %s (p2pkh, p2wpkh or avax)\n", format_name);
        return 1;
    }

    if (lines.empty()) {
        char buf[1024];
        while (fgets(buf, sizeof(buf), stdin)) {
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
                s.pop_back();
            if (!s.empty())
                lines.push_back(s);
        }
    }
    if (lines.empty()) {
        fprintf(stderr, "usage: mnemonic_to_address [-f p2pkh|p2wpkh|avax] [-p passphrase] [-d path] \"<mnemonic>\" ...\n");
        return 1;
    }

    uint32_t indexes[MAX_DEPTH];
    int depth = bip32_parse_path(path, indexes, MAX_DEPTH);
    if (depth < 0) {
        fprintf(stderr, "bad derivation path: %s\n", path);
        return 1;
    }
    int passphrase_len = (int)strlen(passphrase);
    if (passphrase_len > BIP39_MAX_PASSPHRASE) {
        fprintf(stderr, "passphrase longer than %d bytes\n", BIP39_MAX_PASSPHRASE);
        return 1;
    }

    std::vector<Mnemonic> in(lines.size());
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].size() >= MAX_MNEMONIC) {
            fprintf(stderr, "mnemonic longer than %d bytes: %s\n", MAX_MNEMONIC - 1, lines[i].c_str());
            return 1;
        }
        memcpy(in[i].text, lines[i].data(), lines[i].size());
        in[i].len = (int)lines[i].size();
    }

    ec_group_t group;
    ec_group_init_secp256k1(&group);
    CUDA_CHECK(cudaMemcpyToSymbol(c_group, &group, sizeof(group)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_passphrase, passphrase, passphrase_len));
    CUDA_CHECK(cudaMemcpyToSymbol(c_passphrase_len, &passphrase_len, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_path, indexes, sizeof(indexes)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_depth, &depth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_format, &format, sizeof(int)));

    int n = (int)in.size();
    Mnemonic *d_in;
    Result *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(Mnemonic)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Result)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(Mnemonic), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    derive<<<(n + 127) / 128, 128>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    fprintf(stderr, "%d mnemonics in %.1f ms = %.0f per second\n", n, ms, n / (ms / 1000.0));

    std::vector<Result> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(Result), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    for (int i = 0; i < n; i++) {
        if (!out[i].ok)
            printf("derivation failed %s\n", lines[i].c_str());
        else
            printf("%s%s %s\n", prefix, out[i].address, lines[i].c_str());
    }
    return 0;
}
