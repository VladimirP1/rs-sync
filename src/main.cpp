#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>

#include "calibration.hpp"

int main(int argc, char** argv) {
    std::ifstream fs("GoPro_Hero6_2160p_43.json");

    FisheyeCalibration c;
    
    fs >> c;

    std::cout << c.DistortionCoeffs() << std::endl;
}