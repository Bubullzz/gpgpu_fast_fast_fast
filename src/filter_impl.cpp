#include "filter_impl.h"
#include "filter_params.h"

#include "logo.h"

#include <cmath>
#include <cstring>


// Hardcoded parameters

constexpr float MATCH = 25.0f;
constexpr int FRAME_CANDIDATE = 30;


struct rgb {
    uint8_t r, g, b;
};

struct lab {
    float L, a, b;
};

struct xyz {
    float X, Y, Z;
};

struct states {
    lab background;
    rgb rgb_background;
    lab candidate;
    int last_match = -1;
};

struct vector2i {
    int x;
    int y;
};

// This lut table was generated using the following function (values reducted between 0 and 1):
// // SRGB to XZY to Lab conversion
// // All formulas can be found on http://www.brucelindbloom.com/
// float inverse_srgb_companding(float x) {
//     if (x <= 0.04045f)
//         return x / 12.92f;
//     else {
//         return std::powf((x + 0.055f) / 1.055f, 2.4f);
//     }
// }
constexpr float inverse_srgb_lut[256] = {
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
    0.97344529f, 0.98225055f, 0.99110210f, 1.00000000f
};

constexpr float epsilon = 0.008856f;
constexpr float kappa = 903.3f;

// http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html#Specifications
// D65 sRGB illuminant values
static xyz rgb_to_xyz(rgb color) {
    xyz result;

    float x = inverse_srgb_lut[color.r];
    float y = inverse_srgb_lut[color.g];
    float z = inverse_srgb_lut[color.b];

    result.X = x * 0.4124564f + y * 0.3575761f + z * 0.1804375f;
    result.Y = x * 0.2126729f + y * 0.7151522f + z * 0.0721750f;
    result.Z = x * 0.0193339f + y * 0.1191920f + z * 0.9503041f;
    return result;
}

static float lab_f(float t) {
    return (t > epsilon) ? std::cbrt(t) : (kappa * t + 16.0f) / 116.0f;
}

// https://fr.mathworks.com/help/images/ref/whitepoint.html
// D65 illuminant reference white point
constexpr float Xr = 1.0f / 0.95047f; // Reference white point X
constexpr float Yr = 1.0f / 1.00000f; // Reference white point Y
constexpr float Zr = 1.0f / 1.08883f; // Reference white point Z

static lab xyz_to_lab(xyz color) {
    lab result;
    float xr = color.X * Xr;
    float yr = color.Y * Yr;
    float zr = color.Z * Zr;

    float fx = lab_f(xr);
    float fy = lab_f(yr);
    float fz = lab_f(zr);

    result.L = std::max(std::min(116.0f * fy - 16.0f, 100.0f), 0.0f);
    result.a = 500.0f * (fx - fy);
    result.b = 200.0f * (fy - fz);
    return result;
}

// Complete formulas can be found on http://www.brucelindbloom.com/
static inline lab rgb_to_lab(const rgb &color) {
    xyz xyz_color = rgb_to_xyz(color);
    return xyz_to_lab(xyz_color);
}

static float lab_distance(const lab& a, const lab& b) {
    float delta_L = a.L - b.L;
    float delta_a = a.a - b.a;
    float delta_b = a.b - b.b;

    return std::sqrt(delta_L * delta_L + delta_a * delta_a + delta_b * delta_b);
}

static lab lab_lerp(const lab& a, const lab& b, const float coef) {
    const float c1 = 1.0f - coef;
    const float c2 = coef;
    return lab {
        .L = c1 * a.L + c2 * b.L,
        .a = c1 * a.a + c2 * b.a,
        .b = c1 * a.b + c2 * b.b
    };
}

static rgb rgb_lerp(const rgb& a, const rgb& b, const float coef) {
    const float c1 = 1.0f - coef;
    const float c2 = coef;
    return rgb {
        .r = (uint8_t) (c1 * (float) a.r + c2 * (float) b.r),
        .g = (uint8_t) (c1 * (float) a.g + c2 * (float) b.g),
        .b = (uint8_t) (c1 * (float) a.b + c2 * (float) b.b)
    };
}

static inline lab lab_average(const lab& a, const lab& b) {
    return lab_lerp(a, b, 0.5f);
}

static float background_estimation(states& state, const rgb& rgb_pixel) {
    lab lab_pixel = rgb_to_lab(rgb_pixel);

    if (state.last_match < 0) {
        state.background = lab_pixel;
        state.rgb_background = rgb_pixel;
        state.candidate = lab_pixel; // May cause issues?? (not in doc (which doc ?))
        state.last_match = 0;
    }

    //if (match_distance < MATCH) {
    //    state.background = lab_lerp(state.background, lab_pixel, 0.1f);
    //}

    float candidate_distance = lab_distance(state.candidate, lab_pixel);
    if (candidate_distance < 4.0f) {
        state.candidate = lab_lerp(state.candidate, lab_pixel, 0.2f);
        if (state.last_match >= FRAME_CANDIDATE) {
            state.background = state.candidate;
            state.rgb_background = rgb_pixel;
        } else {
            state.last_match++;
        }
    } else {
        state.last_match = 1;
        state.candidate = lab_pixel;
    }

    float match_distance = lab_distance(state.background, lab_pixel);

    /*
    if (match_distance < MATCH) {
        state.last_match = 0;
        state.background = lab_average(state.background, lab_pixel);
    } else {
        if (!state.last_match) {
            state.candidate = lab_pixel;
            state.last_match++;
        } else if (state.last_match < FRAME_CANDIDATE) {
            state.candidate = lab_average(state.candidate, lab_pixel);
            state.last_match++;
        } else {
            auto tmp = state.background;
            state.background = state.candidate;
            state.candidate = tmp;
            state.last_match = 0;
        }
    }
    */

    return match_distance;
}

static void blur(const float* input, float* output, int size, int width, int height, int stride) {
    float weights_sum = 0.0f;
    for (int dy = -size; dy <= size; ++dy) {
        for (int dx = -size; dx <= size; ++dx) {
            // Check if the pixel is inside the kernel (circle shape)
            //if (dx * dx + dy * dy > size * size) continue;
            weights_sum += 1.0f;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum_value = 0.0f;

            for (int dy = -size; dy <= size; ++dy) {
                for (int dx = -size; dx <= size; ++dx) {
                    // Check if the pixel is inside the kernel (circle shape)
                    //if (dx * dx + dy * dy > size * size) continue;

                    // Check if the pixel is inside the image
                    if (y + dy < 0 || y + dy >= height || x + dx < 0 || x + dx >= width) {
                        continue;
                    }

                    const float value = input[(y + dy) * stride + (x + dx)];
                    sum_value += value;
                }
            }

            output[y * stride + x] = sum_value / weights_sum;
        }
    }
}

// Please check if it works
// This Erosion for BINARIZED images ONLY ([0, 255])
// Maybe we would want to adapt for greyscale images
static void erode(const float* input, float* output, int size, int width, int height, int stride) {
    // int kernel_size = 0;
    // for (int dy = -size; dy <= size; ++dy) {
    //     for (int dx = -size; dx <= size; ++dx) {
    //         // Check if the pixel is inside the kernel (circle shape)
    //         //if (dx * dx + dy * dy > size * size) continue;
    //         kernel_size++;
    //     }
    // }

    //const int threshold = kernel_size / 8;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            //int nb_low_pixels = 0;
            float min_value = INFINITY;

            for (int dy = -size; dy <= size; ++dy) {
                for (int dx = -size; dx <= size; ++dx) {
                    // Check if the pixel is inside the kernel (circle shape)
                    //if (dx * dx + dy * dy > size * size) continue;

                    // Check if the pixel is inside the image
                    if (y + dy < 0 || y + dy >= height || x + dx < 0 || x + dx >= width) {
                        continue;
                    }

                    const float value = input[(y + dy) * stride + (x + dx)];
                    if (value < min_value) {
                        min_value = value;
                    }
                }
            }

            output[y * stride + x] = min_value;
        }
    }
}

// Please check if it works
// This Dilation is for BINARIZED images ONLY ([0, 255])
// Maybe we would want to adapt for greyscale images
static void dilate(const float* input, float* output, int size, int width, int height, int stride) {
    // int kernel_size = 0;
    // for (int dy = -size; dy <= size; ++dy) {
    //     for (int dx = -size; dx <= size; ++dx) {
    //         // Check if the pixel is inside the kernel (circle shape)
    //         //if (dx * dx + dy * dy > size * size) continue;
    //         kernel_size++;
    //     }
    // }

    //const int threshold = kernel_size / 8;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            //int nb_high_pixels = 0;
            float max_value = 0.0f;

            for (int dy = -size; dy <= size; ++dy) {
                for (int dx = -size; dx <= size; ++dx) {
                    // Check if the pixel is inside the kernel (circle shape)
                    //if (dx * dx + dy * dy > size * size) continue;

                    // Check if the pixel is inside the image
                    if (y + dy < 0 || y + dy >= height || x + dx < 0 || x + dx >= width) {
                        continue;
                    }

                    const float value = input[(y + dy) * stride + (x + dx)];
                    if (value > max_value) {
                        max_value = value;
                    }
                }
            }

            output[y * stride + x] = max_value;
        }
    }
}


// Starting from seed points, expand to neightboors following a mask
static void seed(
    const float* mask, const float* seeds, float* output,
    int width, int height, int stride
) {
    memcpy(output, seeds, stride * height * sizeof(float));

    constexpr vector2i NEIGHBORS[] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }
    };

    const int max_queue_size = width * height / 2;
    vector2i queue[max_queue_size];
    int queue_size = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (seeds[y * stride + x] > 0.5f) {
                for (const vector2i &offset : NEIGHBORS) {
                    // Get neighbor coordinates
                    const int nx = x + offset.x;
                    const int ny = y + offset.y;

                    // Check if inside image boundaries
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }

                    if (mask[ny * stride + nx] > 0.5f) {
                        output[ny * stride + nx] = 1.0f;
                        queue[queue_size++] = {nx, ny};
                    }
                }
            }
        }
    }

    while (queue_size > 0) {
        const vector2i current = queue[--queue_size];
        for (const vector2i &offset : NEIGHBORS) {
            // Get neighbor coordinates
            const int nx = current.x + offset.x;
            const int ny = current.y + offset.y;

            // Check if inside image boundaries
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            if (mask[ny * stride + nx] > 0.5f && output[ny * stride + nx] == 0.0f) {
                output[ny * stride + nx] = 1.0f;
                queue[queue_size++] = {nx, ny};
            }
        }
    }
}


// Hysteresis seems to be applied for colored images, after dilation and erosion,
// To be checked
// https://developer.imageviz.com/refmans/latest/ImageDev/html/Processing_ImageSegmentation_Binarization_HysteresisThresholding.html
static void hysteresis(
    const float* input, float* output,
    int width, int height, int stride,
    float low_threshold, float high_threshold
) {
    constexpr vector2i NEIGHBORS[] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }
    };

    const int mx = width - 1;
    const int my = height - 1;

    const int max_queue_size = width * height / 2;
    vector2i queue[max_queue_size];
    int queue_size = 0;

    for (int y = 1; y < my; ++y) {
        for (int x = 1; x < mx; ++x) {
            const float value = input[y * stride + x];
            if (value >= high_threshold) {
                output[y * stride + x] = 1.0f;
            } else if (value < low_threshold) {
                output[y * stride + x] = 0.0f;
            } else {
                output[y * stride + x] = 0.5;
                for (const vector2i &offset : NEIGHBORS) {
                    const int nx = x + offset.x;
                    const int ny = y + offset.y;
                    if (input[ny * stride + nx] >= high_threshold) {
                        output[y * stride + x] = 1.0f;
                        queue[queue_size++] = {x, y};
                        break;
                    }
                }
            }
        }
    }

    while (queue_size > 0) {
        const vector2i current = queue[--queue_size];
        for (const vector2i &offset : NEIGHBORS) {
            // Get neighbor coordinates
            const int nx = current.x + offset.x;
            const int ny = current.y + offset.y;

            // Check if inside image boundaries
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            // Check if neighbor is already between low and high thresholds
            if (output[ny * stride + nx] == 0.5f) {
                output[ny * stride + nx] = 1.0f;
                queue[queue_size++] = {nx, ny};
            }
        }
    }

    // Set all unconnected pixels between low and high threshold to 0
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (output[y * stride + x] == 0.5f) {
                output[y * stride + x] = 0.0f;
            }
        }
    }

    // Set output left and right border to 0
    const int last_column_offset = width - 1;
    for (int y = 0; y < height; ++y) {
        output[y * stride] = 0.0f;
        output[y * stride + last_column_offset] = 0.0f;
    }

    // Set output top and bottom border to 0
    const int last_line_offset = (height - 1) * stride;
    for (int x = 0; x < width; ++x) {
        output[x] = 0.0f;
        output[last_line_offset + x] = 0.0f;
    }
}

states* pixel_states = nullptr;
float* last_result = nullptr;

extern "C" void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride, const GstFilterParams params) {
    if (pixel_states == nullptr) {
        pixel_states = new states[width * height];
    }

    if (last_result == nullptr) {
        last_result = new float[width * height]{};
    }

    float buffer_1[width * height];
    float buffer_2[width * height];
    float buffer_3[width * height];

    // Compute movement estimation for each pixel
    for (int y = 0; y < height; ++y) {
        rgb* lineptr = (rgb*)(buffer + y * stride);
        for (int x = 0; x < width; ++x) {
            const float match_distance = background_estimation(pixel_states[y * width + x], lineptr[x]);
            buffer_1[y * width + x] = match_distance; // + last_result[y * width + x] * 5.0f;
        }
    }

    float* render_buffer = buffer_1;

    // hysteresis(buffer_1, buffer_2, width, height, width, 10.0f, 50.0f);
    // erode(buffer_2, buffer_1, 3, width, height, width);
    // dilate(buffer_1, buffer_3, 1, width, height, width);
    // seed(buffer_2, buffer_3, buffer_1, width, height, width);
    // render_buffer = buffer_1;

    // hysteresis(buffer_1, buffer_2, width, height, width, 10.0f, 50.0f);
    // dilate(buffer_2, buffer_1, 2, width, height, width);
    // erode(buffer_1, buffer_3, 5, width, height, width);
    // seed(buffer_2, buffer_3, buffer_1, width, height, width);
    // render_buffer = buffer_1;

    erode(buffer_1, buffer_2, 3, width, height, width);
    dilate(buffer_2, buffer_1, 3, width, height, width);
    hysteresis(buffer_1, buffer_2, width, height, width, 4.0f, 30.0f);
    render_buffer = buffer_2;

    // hysteresis(buffer_1, buffer_2, width, height, width, 10.0f, 50.0f);
    // blur(buffer_2, buffer_3, 3, width, height, width);
    // hysteresis(buffer_3, buffer_1, width, height, width, 0.5f, 0.6f);
    // erode(buffer_1, buffer_3, 4, width, height, width);
    // dilate(buffer_3, buffer_1, 3, width, height, width);
    // seed(buffer_2, buffer_1, buffer_3, width, height, width);
    // render_buffer = buffer_3;

    // New buffer / change how we handle buffers (TODO / SAVE ORIGIN BUFFER AND CREATE NEW MASK /// memcpy??)
    // Greyscale or binarize buffer (for erosion and dilation)
    // Erode and dilate BW BUFFER (MOSTLY DONE, BINARY ONLY SUPPORTED FOR NOW)
    // Hysteresis (TO COMPLETE / CHECK) // Seems that input is colored/grey and output is binary
    // Use BW buffer to mask

    // Render result

    memcpy(last_result, render_buffer, width * height * sizeof(float));

    for (int y = 0; y < height; ++y) {
        rgb* lineptr = (rgb*)(buffer + y * stride);
        for (int x = 0; x < width; ++x) {
            uint8_t intensity = std::min(static_cast<int>(render_buffer[y * width + x] * 255.0f), 255);

            //lineptr[x] = pixel_states[y * width + x].rgb_background;

            const rgb pixel = lineptr[x];

            if (intensity == 0) {
                intensity = pixel.r;
            }

            lineptr[x].r = (pixel.r + intensity * 2) / 3;
            lineptr[x].g = pixel.g;
            lineptr[x].b = pixel.b;
        }
    }
}
