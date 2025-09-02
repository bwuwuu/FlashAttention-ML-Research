#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

__global__ void forward_kernel(const float* Q, const float* K, const float* V,
                    const int N, const int d, const int Br, const int Bc, const int Tr, const int Tc, const float softmax_scale,
                    float* O, float* l, float *m) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int gy = gridDim.y;

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    int qkv_offset = (gy * bx * N * d) + (by * N * d);
    int lm_offset = (gy * bx * N) + (by * N);

    for (int j = 0; j < Tc; j++) {
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)  {
            for (int x = 0; x < d; x++)
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];

            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++)
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            float m_prev = m[lm_offset + (Br * i) + tx];
            float l_prev = l[lm_offset + (Br * i) + tx];

            float m_new = max(m_prev, row_m);
            float l_new = (__expf(m_prev - m_new) * l_prev) + (__expf(row_m - m_new) * row_l);

            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++)
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / l_new) \
                    * ((l_prev * __expf(m_prev - m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = m_new;
            l[lm_offset + (Br * i) + tx] = l_new;
        }
        __syncthreads();
    }
}

torch::Tensor f_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // variable names follow those from the reference paper
    const int B = Q.size(0); // batch_size
    const int nh = Q.size(1); // n_head
    const int N = Q.size(2); // seq_len
    const int d = Q.size(3); // d_model

    const int Br = 32; const int Bc = 32;

    const int Tr = ceil((float) N / Br);
    const int Tc = ceil((float) N / Bc);
    const float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    l = l.to(torch::kCUDA); m = m.to(torch::kCUDA);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    int sram_size = (3 * Bc * d * sizeof(float)) + (Br * Bc * sizeof(float));
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 dimGrid(B, nh);
    dim3 dimBlock(Bc);

    forward_kernel<<<dimGrid, dimBlock, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Br, Bc, Tr, Tc, softmax_scale,
        O.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>()
    );

    return O;
}

