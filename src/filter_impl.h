#pragma once

#include <stdint.h>
#include <stddef.h>

#include "filter_params.h"

typedef struct _GstFilterParams GstFilterParams;

#ifdef __cplusplus
extern "C" {
#endif

void filter_impl(uint8_t* buffer, int width, int height, int plane_stride, int pixel_stride, const GstFilterParams params);

#ifdef __cplusplus
}
#endif
