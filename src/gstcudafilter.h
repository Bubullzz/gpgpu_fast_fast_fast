/* GStreamer
 * Copyright (C) 2023 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_CUDA_FILTER_H_
#define _GST_CUDA_FILTER_H_

#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

#include "filter_params.h"

G_BEGIN_DECLS

#define GST_PLUGIN_NAME cudafilter
#define GST_PIPELINE_ELEMENT_NAME "cudafilter"

#define GST_TYPE_CUDA_FILTER   (gst_cuda_filter_get_type())
#define GST_CUDA_FILTER(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CUDA_FILTER,GstCudaFilter))
#define GST_CUDA_FILTER_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CUDA_FILTER,GstCudaFilterClass))
#define GST_IS_CUDA_FILTER(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CUDA_FILTER))
#define GST_IS_CUDA_FILTER_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CUDA_FILTER))

typedef struct _GstCudaFilter GstCudaFilter;
typedef struct _GstCudaFilterClass GstCudaFilterClass;

struct _GstCudaFilter
{
  GstVideoFilter base_cudafilter;

  GstFilterParams params;
};

struct _GstCudaFilterClass
{
  GstVideoFilterClass base_cudafilter_class;
};

GType gst_cuda_filter_get_type (void);

G_END_DECLS

#endif
