extern "C"
{
#include <stdint.h>
}

static __device__ __forceinline__ uint16_t to10(uint8_t v)
{
    return (uint16_t)((v * 1023 + 127) / 255);
}

extern "C" __global__ void kernel_uyvy_to_yuv422p10le(const uint8_t *__restrict__ src,
                                                      int src_pitch, int src_w, int src_h,
                                                      uint16_t *__restrict__ dst_y, int dst_y_pitch,
                                                      uint16_t *__restrict__ dst_u, int dst_u_pitch,
                                                      uint16_t *__restrict__ dst_v, int dst_v_pitch)
{
    int x_pair = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= src_h || x_pair >= (src_w >> 1))
        return;

    const uint8_t *row = src + y * src_pitch;
    int off = (x_pair << 2);

    uint8_t U8 = row[off + 0];
    uint8_t Y0 = row[off + 1];
    uint8_t V8 = row[off + 2];
    uint8_t Y1 = row[off + 3];

    uint16_t Y0_10 = to10(Y0);
    uint16_t Y1_10 = to10(Y1);
    uint16_t U_10 = to10(U8);
    uint16_t V_10 = to10(V8);

    uint16_t *yrow = (uint16_t *)((uint8_t *)dst_y + y * dst_y_pitch);
    uint16_t *urow = (uint16_t *)((uint8_t *)dst_u + y * dst_u_pitch);
    uint16_t *vrow = (uint16_t *)((uint8_t *)dst_v + y * dst_v_pitch);

    int x0 = (x_pair << 1);
    yrow[x0 + 0] = Y0_10;
    yrow[x0 + 1] = Y1_10;

    urow[x_pair] = U_10;
    vrow[x_pair] = V_10;
}

extern "C" int ff_cuda_uyvy_to_yuv422p10le_launch(void *cuda_ctx_opaque, void *cu_stream,
                                                  const uint8_t *src, int src_pitch, int src_w, int src_h,
                                                  uint16_t *dst_y, int dst_y_pitch,
                                                  uint16_t *dst_u, int dst_u_pitch,
                                                  uint16_t *dst_v, int dst_v_pitch)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cu_stream);

    dim3 block(32, 16);
    dim3 grid(((src_w >> 1) + block.x - 1) / block.x,
              (src_h + block.y - 1) / block.y);

    kernel_uyvy_to_yuv422p10le<<<grid, block, 0, stream>>>(
        src, src_pitch, src_w, src_h,
        dst_y, dst_y_pitch,
        dst_u, dst_u_pitch,
        dst_v, dst_v_pitch);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}