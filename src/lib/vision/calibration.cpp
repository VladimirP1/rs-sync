#include "calibration.hpp"

#include <iostream>

#include <nlohmann/json.hpp>

FisheyeCalibration::FisheyeCalibration() {}

std::string FisheyeCalibration::Name() const { return name_; }

double FisheyeCalibration::Rmse() const { return rmse_; }

int FisheyeCalibration::Width() const { return width_; }

int FisheyeCalibration::Height() const { return height_; }

cv::Mat FisheyeCalibration::CameraMatrix() const { return cameraMatrix_; }

cv::Mat FisheyeCalibration::DistortionCoeffs() const { return distCoeffs_; }

bool FisheyeCalibration::IsLoaded() const { return isLoaded_; }

std::istream& operator>>(std::istream& s, FisheyeCalibration& calibration) {
    nlohmann::json json;
    s >> json;

    std::string tmp_name;
    double tmp_rmse{0};
    int tmp_width{0}, tmp_height{0};
    cv::Mat tmp_cameraMatrix{cv::Mat::eye(3, 3, CV_64F)};
    cv::Mat tmp_distCoeffs{cv::Mat::zeros(4, 1, CV_64F)};

    {
        const auto name = json["name"];

        if (!name.is_string()) {
            throw std::runtime_error{"name missing from json"};
        }

        tmp_name = name;
    }

    {
        const auto dim = json["calib_dimension"];

        if (!dim.is_object()) {
            throw std::runtime_error{"calib_dimension missing from json"};
        }

        if (!dim["w"].is_number() || !dim["h"].is_number()) {
            throw std::runtime_error{"calib_dimension wrong format"};
        }

        tmp_width = dim["w"];
        tmp_height = dim["h"];
    }

    {
        const auto fisheye = json["fisheye_params"];

        if (!fisheye.is_object()) {
            throw std::runtime_error{"fisheye_params missing from json"};
        }

        {
            const auto rmse = fisheye["RMS_error"];

            if (!rmse.is_number()) {
                throw std::runtime_error{
                    "fisheye_params.rmse missing from json"};
            }

            tmp_rmse = rmse;
        }

        {
            const auto camera_matrix = fisheye["camera_matrix"];

            if (!camera_matrix.is_array() || camera_matrix.size() != 3) {
                throw std::runtime_error{
                    "fisheye_params.camera_matrix missing from json or "
                    "invalid"};
            }

            int row_idx{0};
            for (const auto& row : camera_matrix) {
                if (!row.is_array() || row.size() != 3) {
                    throw std::runtime_error{
                        "fisheye_params.camera_matrix missing from json or "
                        "invalid"};
                }
                int col_idx{0};
                for (const auto& coeff : row) {
                    if (!coeff.is_number()) {
                        throw std::runtime_error{
                            "fisheye_params.camera_matrix missing from json or "
                            "invalid"};
                    }

                    tmp_cameraMatrix.at<double>(row_idx, col_idx++) = coeff;
                }
                ++row_idx;
            }
        }

        {
            const auto dist_coeffs = fisheye["distortion_coeffs"];

            if (!dist_coeffs.is_array()) {
                throw std::runtime_error{
                    "fisheye_params.distortion_coeffs missing from json"};
            }

            int idx{0};
            for (const auto& coeff : dist_coeffs) {
                tmp_distCoeffs.at<double>(idx++) = coeff;
            }
        }

        calibration.name_ = tmp_name;
        calibration.rmse_ = tmp_rmse;
        calibration.width_ = tmp_width;
        calibration.height_ = tmp_height;
        calibration.cameraMatrix_ = tmp_cameraMatrix;
        calibration.distCoeffs_ = tmp_distCoeffs;
        calibration.isLoaded_ = true;
    }

    return s;
}