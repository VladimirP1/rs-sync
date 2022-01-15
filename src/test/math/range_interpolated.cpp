#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <numeric>

#include <math/prefix_sums.hpp>
#include <math/range_interpolated.hpp>
#include <math/segment_tree.hpp>
#include <math/simple_math.hpp>

TEMPLATE_TEST_CASE("Interpolated<T> queries with different underlying data structures", "",
                   PrefixSums<DefaultGroup<double>>, SegmentTree<DefaultGroup<double>>) {
    using TestedType = Interpolated<TestType>;
    SECTION("Can be created with no data") {
        TestedType t;
        REQUIRE(t.SoftQuery(-.1, .1) == 0);
    }
    SECTION("Can be created on empty data") {
        std::vector<double> data;
        TestedType t{data.begin(), data.end()};
        REQUIRE(t.SoftQuery(0, 0) == 0);
        REQUIRE(t.SoftQuery(0, .1) == 0);
        REQUIRE(t.SoftQuery(-.1, 0) == 0);
        REQUIRE(t.SoftQuery(-5, 5) == 0);
    }

    SECTION("Correctly sums sequences of ones") {
        std::vector<double> data(101, 1.);
        TestedType t{data.begin(), data.end()};
        REQUIRE(t.SoftQuery(0, 1.) == Approx(1.));
        REQUIRE(t.SoftQuery(0, .1) == Approx(.1));
        REQUIRE(t.SoftQuery(.9, 1.1) == Approx(.2));
        REQUIRE(t.SoftQuery(.9, 2.1) == Approx(1.2));
        REQUIRE(t.SoftQuery(.9, 50) == Approx(49.1));
        REQUIRE(t.SoftQuery(.9, 50.9) == Approx(50));
        REQUIRE(t.SoftQuery(1, 50.9) == Approx(49.9));
    }

    SECTION("Correctly sums complex sequences") {
        std::vector<double> data{3, 4, 1, 2, 4, 6, 8, 4, 3};
        TestedType t{data.begin(), data.end()};
        REQUIRE(t.SoftQuery(2.2, 6.7) == Approx(1 * .8 + 2 + 4 + 6 + 8 * .7));
    }
}

TEMPLATE_TEST_CASE("Interpolated<T> does large queries correctly", "",
                   PrefixSums<DefaultGroup<uint8_t>>, SegmentTree<DefaultGroup<uint8_t>>) {
    using TestedType = Interpolated<TestType>;
    SECTION("Correctly sums big sequences") {
        std::vector<uint8_t> data(100'000'000, 255);
        TestedType t{data.begin(), data.end()};

        REQUIRE(t.SoftQuery(0, 100'000'000) ==
                std::accumulate(data.begin(), data.end(), static_cast<uint8_t>(0)));
    }
}