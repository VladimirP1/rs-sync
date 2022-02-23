#pragma once

#include <stddef.h>

extern "C" {
struct tp_gyrodata {
    size_t samples;
    double* timestamps;
    double* gyro;
};

tp_gyrodata tp_load_gyro(const char* path, const char* orient);
}