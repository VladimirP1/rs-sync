#pragma once

#include <chrono>
#include <iostream>
#include <string>

struct Stopwatch {
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    Stopwatch(std::string description = "") : descripiton_(description) {}
    ~Stopwatch() {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> dur = (end - start);
        std::cout << "\"" << descripiton_ << "\""
                  << " took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(dur)
                         .count()
                  << " ms" << std::endl;
    }

   private:
    std::string descripiton_;
};