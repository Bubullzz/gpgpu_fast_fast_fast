#ifndef FILTER_PARAMS_H
#define FILTER_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

// Utiliser types standard C/C++ ici pour éviter les conflits avec CUDA
typedef struct _GstFilterParams {
  const char *bg;
  int opening_size;
  int th_low;
  int th_high;
} GstFilterParams;

#ifdef __cplusplus
}
#endif

#endif // FILTER_PARAMS_H
