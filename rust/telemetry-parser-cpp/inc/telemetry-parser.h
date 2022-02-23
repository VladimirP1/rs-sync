#pragma once

#include <stddef.h>
#include <malloc.h>

extern "C" {
struct tp_gyrodata {
    size_t samples;
    double* timestamps;
    double* gyro;
};

tp_gyrodata tp_load_gyro(const char* path, const char* orient);

inline void tp_free(tp_gyrodata d) {
    if (d.timestamps) free(d.timestamps);
    if (d.gyro) free(d.gyro);
}
}