#include "filter_impl.h"
#include "filter_params.h"

#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include <c++/13/functional>
#include <c++/13/functional>
#include <c++/13/functional>
#include <c++/13/bits/algorithmfwd.h>
#include <nvToolsExt.h>

#include "logo.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
#define TILE_WIDTH 16

#define HYTERESIS_HIGH 30
#define HYTERESIS_LOW 4
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb {
    uint8_t r, g, b;
};

struct lab {
    float l, a, b;
};

struct mask_infos {
    lab bg;
    lab candidate;
    unsigned long time;
};

__constant__ int d_disk_radius;
__constant__ int d_loaded_width;
int h_disk_radius;
int h_loaded_width;

bool first_frame = true;
mask_infos* mask;
uint8_t* mask_output;
uint8_t* erosion_output;
uint8_t* dilatation_output;
bool* hysteresis_output;

size_t mask_pitch = 0;
size_t mask_output_pitch = 0;
size_t erosion_output_pitch = 0;
size_t dilatation_output_pitch = 0;
size_t hysteresis_output_pitch = 0;

__device__ constexpr float inverse_srgb_lut[256] = {
    0.00000000f, 0.00030353f, 0.00060705f, 0.00091058f, 0.00121411f, 0.00151763f, 0.00182116f, 0.00212469f, 0.00242822f,
    0.00273174f, 0.00303527f, 0.00334654f, 0.00367651f, 0.00402472f, 0.00439144f, 0.00477695f, 0.00518152f, 0.00560539f,
    0.00604883f, 0.00651209f, 0.00699541f, 0.00749903f, 0.00802319f, 0.00856813f, 0.00913406f, 0.00972122f, 0.01032982f,
    0.01096009f, 0.01161225f, 0.01228649f, 0.01298303f, 0.01370208f, 0.01444384f, 0.01520851f, 0.01599629f, 0.01680738f,
    0.01764195f, 0.01850022f, 0.01938236f, 0.02028856f, 0.02121901f, 0.02217388f, 0.02315337f, 0.02415763f, 0.02518686f,
    0.02624122f, 0.02732089f, 0.02842604f, 0.02955683f, 0.03071344f, 0.03189603f, 0.03310477f, 0.03433981f, 0.03560131f,
    0.03688945f, 0.03820437f, 0.03954624f, 0.04091520f, 0.04231141f, 0.04373503f, 0.04518620f, 0.04666509f, 0.04817182f,
    0.04970657f, 0.05126946f, 0.05286065f, 0.05448028f, 0.05612849f, 0.05780543f, 0.05951124f, 0.06124605f, 0.06301002f,
    0.06480327f, 0.06662594f, 0.06847817f, 0.07036010f, 0.07227185f, 0.07421357f, 0.07618538f, 0.07818742f, 0.08021982f,
    0.08228271f, 0.08437621f, 0.08650046f, 0.08865559f, 0.09084171f, 0.09305896f, 0.09530747f, 0.09758735f, 0.09989873f,
    0.10224173f, 0.10461648f, 0.10702310f, 0.10946171f, 0.11193243f, 0.11443537f, 0.11697067f, 0.11953843f, 0.12213877f,
    0.12477182f, 0.12743768f, 0.13013648f, 0.13286832f, 0.13563333f, 0.13843162f, 0.14126329f, 0.14412847f, 0.14702727f,
    0.14995979f, 0.15292615f, 0.15592646f, 0.15896084f, 0.16202938f, 0.16513219f, 0.16826940f, 0.17144110f, 0.17464740f,
    0.17788842f, 0.18116424f, 0.18447499f, 0.18782077f, 0.19120168f, 0.19461783f, 0.19806932f, 0.20155625f, 0.20507874f,
    0.20863687f, 0.21223076f, 0.21586050f, 0.21952620f, 0.22322796f, 0.22696587f, 0.23074005f, 0.23455058f, 0.23839757f,
    0.24228112f, 0.24620133f, 0.25015828f, 0.25415209f, 0.25818285f, 0.26225066f, 0.26635560f, 0.27049779f, 0.27467731f,
    0.27889426f, 0.28314874f, 0.28744084f, 0.29177065f, 0.29613827f, 0.30054379f, 0.30498731f, 0.30946892f, 0.31398871f,
    0.31854678f, 0.32314321f, 0.32777810f, 0.33245154f, 0.33716362f, 0.34191442f, 0.34670406f, 0.35153260f, 0.35640014f,
    0.36130678f, 0.36625260f, 0.37123768f, 0.37626212f, 0.38132601f, 0.38642943f, 0.39157248f, 0.39675523f, 0.40197778f,
    0.40724021f, 0.41254261f, 0.41788507f, 0.42326767f, 0.42869050f, 0.43415364f, 0.43965717f, 0.44520119f, 0.45078578f,
    0.45641102f, 0.46207700f, 0.46778380f, 0.47353150f, 0.47932018f, 0.48514994f, 0.49102085f, 0.49693300f, 0.50288646f,
    0.50888132f, 0.51491767f, 0.52099557f, 0.52711513f, 0.53327640f, 0.53947949f, 0.54572446f, 0.55201140f, 0.55834039f,
    0.56471151f, 0.57112483f, 0.57758044f, 0.58407842f, 0.59061884f, 0.59720179f, 0.60382734f, 0.61049557f, 0.61720656f,
    0.62396039f, 0.63075714f, 0.63759687f, 0.64447968f, 0.65140564f, 0.65837482f, 0.66538730f, 0.67244316f, 0.67954247f,
    0.68668531f, 0.69387176f, 0.70110189f, 0.70837578f, 0.71569350f, 0.72305513f, 0.73046074f, 0.73791041f, 0.74540421f,
    0.75294222f, 0.76052450f, 0.76815115f, 0.77582222f, 0.78353779f, 0.79129794f, 0.79910274f, 0.80695226f, 0.81484657f,
    0.82278575f, 0.83076988f, 0.83879901f, 0.84687323f, 0.85499261f, 0.86315721f, 0.87136712f, 0.87962240f, 0.88792312f,
    0.89626935f, 0.90466117f, 0.91309865f, 0.92158186f, 0.93011086f, 0.93868573f, 0.94730654f, 0.95597335f, 0.96468625f,
    0.97344529f, 0.98225055f, 0.99110210f, 1.00000000f};


constexpr float epsilon = 0.008856f;
constexpr float kappa = 903.3f;
__device__ inline float lab_f(float t) {
    return (t > epsilon) ? std::cbrt(t) : (kappa * t + 16.0f) / 116.0f;
}


__global__ void rgb_to_lab(rgb* rgb_buff, lab* lab_buff, int width, int height, int rgb_pitch, int lab_pitch)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* rgb_col = &((rgb*)((std::byte*)rgb_buff + y * rgb_pitch))[x];
    lab* lab_col = &((lab*)((std::byte*)lab_buff + y * lab_pitch))[x];
    float a = inverse_srgb_lut[rgb_col->r];
    float b = inverse_srgb_lut[rgb_col->g];
    float c = inverse_srgb_lut[rgb_col->b];

    float v1 = a * 0.4124564f + b * 0.3575761f + c * 0.1804375f;
    float v2 = a * 0.2126729f + b * 0.7151522f + c * 0.0721750f;
    float v3 = a * 0.0193339f + b * 0.1191920f + c * 0.9503041f;

    constexpr float Xr = 1.0f / 0.95047f; // Reference white point X
    constexpr float Yr = 1.0f / 1.00000f; // Reference white point Y
    constexpr float Zr = 1.0f / 1.08883f; // Reference white point Z

    float xr = v1 * Xr;
    float yr = v2 * Yr;
    float zr = v3 * Zr;

    float fx = lab_f(xr);
    float fy = lab_f(yr);
    float fz = lab_f(zr);

    float L = max(min(116.0f * fy - 16.0f, 100.0f), 0.0f);
    float A = 500.0f * (fx - fy);
    float B = 200.0f * (fy - fz);

    lab_col->l = L;
    lab_col->a = A;
    lab_col->b = B;
}
__global__ void setup_mask_first_frame(lab* first_frame, mask_infos* mask, int first_frame_pitch, int mask_pitch, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    lab* curr_lab = &((lab*)((std::byte*)first_frame + y * first_frame_pitch))[x];
    mask_infos* curr_mask= &((mask_infos*)((std::byte*)mask + y * mask_pitch))[x];

    curr_mask->bg = *curr_lab;
    curr_mask->candidate = *curr_lab;
    curr_mask->time = 0;
}

__global__ void reset_hysteresis(bool* hysteresis_buffer, int hysteresis_pitch, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width || y >= height)
        return;
    bool* curr_hyst = &((bool*)((std::byte*)hysteresis_buffer + y * hysteresis_pitch))[x];
    *curr_hyst = false;
}

__global__ void update_mask(lab* lab_frame, mask_infos* mask, uint8_t* mask_output, int lab_frame_pitch, size_t mask_pitch, size_t mask_output_pitch, int width, int height) {
    int GHOSTING = 50;

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    lab* curr_lab = &((lab*)((std::byte*)lab_frame + y * lab_frame_pitch))[x];
    mask_infos* curr_mask= &((mask_infos*)((std::byte*)mask + y * mask_pitch))[x];

    int dl = curr_lab->l - curr_mask->bg.l;
    int da = curr_lab->a - curr_mask->bg.a;
    int db = curr_lab->b - curr_mask->bg.b;

    int tot_dist = sqrtf(dl * dl + da * da + db * db);
    if (tot_dist >= 25) { // we do not have a match
        if (curr_mask->time == 0) {
            curr_mask->candidate = *curr_lab;
            curr_mask->time += 1;
        }
        else if (curr_mask->time < GHOSTING) {
            curr_mask->candidate.l /= 2;
            curr_mask->candidate.a /= 2;
            curr_mask->candidate.b /= 2;
            curr_mask->candidate.l += curr_lab->l / 2;
            curr_mask->candidate.a += curr_lab->a / 2;
            curr_mask->candidate.b += curr_lab->b / 2;

            curr_mask->time += 1;
        }
        else {
            lab tmp = curr_mask->bg;
            curr_mask->bg = curr_mask->candidate;
            curr_mask->candidate = tmp;
            curr_mask->time = 0;
        }
    }
    else {
        curr_mask->bg.l /= 2;
        curr_mask->bg.a /= 2;
        curr_mask->bg.b /= 2;
        curr_mask->bg.l += curr_lab->l / 2;
        curr_mask->bg.a += curr_lab->a / 2;
        curr_mask->bg.b += curr_lab->b / 2;
        curr_mask->time = 0;
    }
    uint8_t* curr_output = &((uint8_t*)((std::byte*)mask_output + y * mask_output_pitch))[x];
    *curr_output = tot_dist;
}

__global__ void display_mask(rgb* buff, bool* hysteresis_output, int buff_pitch, size_t hysteresis_output_pitch, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* curr_rgb = &((rgb*)((std::byte*)buff + y * buff_pitch))[x];
    bool* curr_hyst= &((bool*)((std::byte*)hysteresis_output + y * hysteresis_output_pitch))[x];
    if (*curr_hyst) {
        curr_rgb->r = (int)curr_rgb->r + 128 > 255 ? 255 : curr_rgb->r + 128;
    }
}

__global__ void erosion(uint8_t* mask_output, uint8_t* erosion_output, size_t mask_output_pitch, size_t erosion_output_pitch, int width, int height) {
    extern __shared__ uint8_t s_distances_erosion[];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int block_y = blockIdx.y * blockDim.y;
    int block_x = blockIdx.x * blockDim.x;

    // LOADING SHARED MEMORY
    int src = threadIdx.y * blockDim.x + threadIdx.x;
    while (src < d_loaded_width * d_loaded_width) {
        int dest_y = src / d_loaded_width;
        int dest_x = src % d_loaded_width;
        src += blockDim.y * blockDim.x;
        int pos_y = block_y - d_disk_radius + dest_y;
        int pos_x = block_x - d_disk_radius + dest_x;
        if (pos_x < 0 || pos_y < 0 || pos_x >= width || pos_y >= height)
            s_distances_erosion[dest_y * d_loaded_width + dest_x] = 255;
        else {
            uint8_t* curr_mask = &((uint8_t*)((std::byte*)mask_output + pos_y * mask_output_pitch))[pos_x];
            s_distances_erosion[dest_y * d_loaded_width + dest_x] = *curr_mask;
        }
    }
    __syncthreads();

    if (x >= width || y >= height)
        return;

    // EROSION
    uint8_t min = 255;
    for (int i = -d_disk_radius; i <= d_disk_radius; i++) {
        for (int j = -d_disk_radius; j <= d_disk_radius; j++) {
            if (i*i + j*j > d_disk_radius * d_disk_radius) continue;
            int sy = threadIdx.y + j + d_disk_radius;
            int sx = threadIdx.x + i + d_disk_radius;
            uint8_t curr = s_distances_erosion[sy * d_loaded_width + sx];
            if (curr < min)
                min = curr;
        }
    }

    __syncthreads();

    uint8_t* curr_erosion = &((uint8_t*)((std::byte*)erosion_output + y * erosion_output_pitch))[x];
    *curr_erosion = min;
}

__global__ void dilatation(uint8_t* erosion_output, uint8_t* dilatation_output, size_t erosion_output_pitch, size_t dilatation_output_pitch, int width, int height) {
    extern __shared__ uint8_t s_distances_dilatation[];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int block_y = blockIdx.y * blockDim.y;
    int block_x = blockIdx.x * blockDim.x;

    // LOADING SHARED MEMORY
    int src = threadIdx.y * blockDim.x + threadIdx.x;
    while (src < d_loaded_width * d_loaded_width) {
        int dest_y = src / d_loaded_width;
        int dest_x = src % d_loaded_width;
        src += blockDim.y * blockDim.x;
        int pos_y = block_y - d_disk_radius + dest_y;
        int pos_x = block_x - d_disk_radius + dest_x;
        if (pos_x < 0 || pos_y < 0 || pos_x >= width || pos_y >= height) {
            s_distances_dilatation[dest_y * d_loaded_width + dest_x] = 0;
            continue; // On the border
        }
        uint8_t* curr_erosion = &((uint8_t*)((std::byte*)erosion_output + pos_y * erosion_output_pitch))[pos_x];
        s_distances_dilatation[dest_y * d_loaded_width + dest_x] = *curr_erosion;
    }
    __syncthreads();

    if (x >= width || y >= height)
    return;

    // DILATATION
    uint8_t max = 0;
    for (int i = -d_disk_radius; i <= d_disk_radius; i++) {
        for (int j = -d_disk_radius; j <= d_disk_radius; j++) {
            if (i*i + j*j > d_disk_radius * d_disk_radius) continue;
            int sy = threadIdx.y + j + d_disk_radius;
            int sx = threadIdx.x + i + d_disk_radius;
            if (i*i + j*j > d_disk_radius * d_disk_radius) continue;
            uint8_t curr = s_distances_dilatation[sy * d_loaded_width + sx];
            max = curr > max ? curr : max;
        }
    }
    uint8_t* curr_dilatation_output = &((uint8_t*)((std::byte*)dilatation_output + y * dilatation_output_pitch))[x];
    *curr_dilatation_output = max;
}

__global__ void hysteresis(uint8_t* dilatation_output, bool* hysteresis_output, size_t dilatation_output_pitch, size_t hysteresis_output_pitch, int width, int height, bool* changed) {
    extern __shared__ bool s_hysteresis[];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int block_y = blockIdx.y * blockDim.y;
    int block_x = blockIdx.x * blockDim.x;

    // LOADING SHARED MEMORY
    int src = threadIdx.y * blockDim.x + threadIdx.x;
    while (src < d_loaded_width * d_loaded_width) {
        int dest_y = src / d_loaded_width;
        int dest_x = src % d_loaded_width;
        src += blockDim.y * blockDim.x;
        int pos_y = block_y - d_disk_radius + dest_y;
        int pos_x = block_x - d_disk_radius + dest_x;
        if (pos_x < 0 || pos_y < 0 || pos_x >= width || pos_y >= height) {
            s_hysteresis[dest_y * d_loaded_width + dest_x] = false;
            continue; // On the border
        }
        bool* curr_hysteresis_output= &((bool*)((std::byte*)hysteresis_output + pos_y * hysteresis_output_pitch))[pos_x];

        s_hysteresis[dest_y * d_loaded_width + dest_x] = *curr_hysteresis_output;
    }
    __syncthreads();

    if (x >= width || y >= height)
        return;

    // HYSTERESIS
    uint8_t* curr_dilatation_output = &((uint8_t*)((std::byte*)dilatation_output + y * dilatation_output_pitch))[x];
    bool* curr_hysteresis_output= &((bool*)((std::byte*)hysteresis_output + y * hysteresis_output_pitch))[x];
    if (*curr_hysteresis_output || *curr_dilatation_output < HYTERESIS_LOW) // Already computed or under thershold
        return;

    if (*curr_dilatation_output > HYTERESIS_HIGH) {
        *curr_hysteresis_output = true;
        *changed = true;
    }
    // Computing doubt values
    for (int i = -d_disk_radius; i <= d_disk_radius; i++) {
        for (int j = -d_disk_radius; j <= d_disk_radius; j++) {
            if (i*i + j*j > d_disk_radius * d_disk_radius) continue;
            if (s_hysteresis[(threadIdx.y + j + d_disk_radius) * d_loaded_width + (threadIdx.x + i + d_disk_radius)]) {
                *curr_hysteresis_output = true;
                *changed = true;
            }
        }
    }
}


extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_pitch, int pixel_stride, GstFilterParams params)
    {
        assert(sizeof(rgb) == pixel_stride);
        rgb* d_rgb_buffer;
        size_t rgb_pitch;

        lab* d_lab_buffer;
        size_t lab_pitch;

        cudaError_t err;
        
        err = cudaMallocPitch(&d_rgb_buffer, &rgb_pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(d_rgb_buffer, rgb_pitch, src_buffer, src_pitch, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        err = cudaMallocPitch(&d_lab_buffer, &lab_pitch, width * sizeof(lab), height);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(TILE_WIDTH,TILE_WIDTH);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        nvtxRangePushA("rgb_to_lab");
        rgb_to_lab<<<gridSize, blockSize>>>(d_rgb_buffer, d_lab_buffer, width, height, rgb_pitch, lab_pitch);
        cudaDeviceSynchronize();
        nvtxRangePop();
        err = cudaGetLastError(); // Get launch error
        CHECK_CUDA_ERROR(err);

        if (first_frame) {
            first_frame = false;

            // Setting up disk_radius and loaded_width
            h_disk_radius = width / 100;
            if (params.opening_size != -1)
            {
                h_disk_radius = params.opening_size;
            }
            cudaMemcpyToSymbol(d_disk_radius, &h_disk_radius, sizeof(int));

            h_loaded_width = TILE_WIDTH + 2 * h_disk_radius;
            cudaMemcpyToSymbol(d_loaded_width, &h_loaded_width, sizeof(int));

            err = cudaMallocPitch(&mask, &mask_pitch, width * sizeof(mask_infos), height);
            CHECK_CUDA_ERROR(err);
            err = cudaMallocPitch(&mask_output, &mask_output_pitch, width * sizeof(uint8_t), height);
            CHECK_CUDA_ERROR(err);
            err = cudaMallocPitch(&erosion_output, &erosion_output_pitch, width * sizeof(uint8_t), height);
            CHECK_CUDA_ERROR(err);
            err = cudaMallocPitch(&dilatation_output, &dilatation_output_pitch, width * sizeof(uint8_t), height);
            CHECK_CUDA_ERROR(err);
            err = cudaMallocPitch(&hysteresis_output, &hysteresis_output_pitch, width * sizeof(bool), height);
            CHECK_CUDA_ERROR(err);
            setup_mask_first_frame<<<gridSize, blockSize>>>(d_lab_buffer, mask, lab_pitch, mask_pitch, width, height);
            cudaDeviceSynchronize();
            err = cudaGetLastError(); // Get launch error
            CHECK_CUDA_ERROR(err);
        }
        nvtxRangePushA("reset_hysteresis");
        reset_hysteresis<<<gridSize, blockSize>>>(hysteresis_output, hysteresis_output_pitch, width, height);
        cudaDeviceSynchronize();
        nvtxRangePop();

        nvtxRangePushA("update_mask");
        update_mask<<<gridSize, blockSize>>>(d_lab_buffer, mask, mask_output, lab_pitch, mask_pitch, mask_output_pitch, width, height);
        cudaDeviceSynchronize();
        nvtxRangePop();
        err = cudaGetLastError(); // Get launch error
        CHECK_CUDA_ERROR(err);

        nvtxRangePush("erosion");
        erosion<<<gridSize, blockSize, sizeof(uint8_t) * h_loaded_width * h_loaded_width>>>(mask_output, erosion_output, mask_output_pitch, erosion_output_pitch, width, height);
        cudaDeviceSynchronize();
        nvtxRangePop();

        nvtxRangePush("dilatation");
        dilatation<<<gridSize, blockSize, sizeof(uint8_t) * h_loaded_width * h_loaded_width>>>(erosion_output, dilatation_output, erosion_output_pitch, dilatation_output_pitch, width, height);
        cudaDeviceSynchronize();
        nvtxRangePop();
        err = cudaGetLastError(); // Get launch error
        CHECK_CUDA_ERROR(err);

        bool h_change = true;
        bool* d_change;
        cudaMalloc(&d_change, sizeof(bool));
        while (h_change) {
            cudaMemset(d_change, false, sizeof(bool));
            nvtxRangePush("hysteresis");
            hysteresis<<<gridSize, blockSize, h_loaded_width * h_loaded_width>>>(dilatation_output, hysteresis_output, dilatation_output_pitch, hysteresis_output_pitch, width, height, d_change);
            cudaDeviceSynchronize();
            nvtxRangePop();
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaMemcpy(&h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost);
        }

        nvtxRangePush("display_mask");
        display_mask<<<gridSize, blockSize>>>(d_rgb_buffer, hysteresis_output, rgb_pitch, hysteresis_output_pitch, width, height);
        cudaDeviceSynchronize();
        nvtxRangePop();
        err = cudaGetLastError(); // Get launch error
        CHECK_CUDA_ERROR(err);

        // Final Copy
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(src_buffer, src_pitch, d_rgb_buffer, rgb_pitch, width * sizeof(rgb), height, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);

        cudaFree(d_rgb_buffer);
        cudaFree(d_lab_buffer);
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }
}
