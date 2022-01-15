#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <ds/sync_context.hpp>

TEST_CASE("SyncContext") {
    SyncContext ctx;

    SECTION("Frames can be accessed") {
        auto num_from_context = ctx.InFrameContext<int>(
            0, [](FrameContext& frame_ctx, int x) { return x; }, 42);
        REQUIRE(num_from_context == 42);
    }

    SECTION("Frames can be accessed const") {
        ctx.InConstFrameContext<void>(0, [](const FrameContext&) {});
    }

    SECTION("Corners can be set inside a frame") {
        std::vector<cv::Point2d> corners{{1, 1}};
        ctx.InFrameContext<void>(
            1,
            [](FrameContext& frame_ctx, std::vector<cv::Point2d>& corners_arg) {
                frame_ctx.SetCorners(corners_arg, 0.);
            },
            corners);

        auto corners2 = ctx.InFrameContext<std::vector<cv::Point2d>>(
            1,
            [](FrameContext& frame_ctx, std::vector<cv::Point2d>& corners_arg) {
                return frame_ctx.Corners();
            },
            corners);

        REQUIRE(std::equal(corners.begin(), corners.end(), corners2.begin()));
    }

    SECTION("Gyro data can be set") {
        std::vector<Quaternion> data{100, Quaternion{0, 0, 1}};
        ctx.SetGyroData(data.begin(), data.end());
    }

    SECTION("Gyro integrator returns reasonable answers") {
        double x, y, z;
        std::vector<Quaternion> data{100, Quaternion{0, 0, 1}};
        ctx.SetGyroData(data.begin(), data.end());

        auto quat = ctx.IntegrateGyro(1.5, 2.5);
        quat.ToRotVec(x, y, z);
        REQUIRE(x == Approx(0));
        REQUIRE(y == Approx(0));
        REQUIRE(z == Approx(1));
    }
}