#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <math/prefix_sums.hpp>
#include <math/range_interpolated.hpp>
#include <math/segment_tree.hpp>
#include <math/simple_math.hpp>

// TEMPLATE_TEST_CASE_SIG("TemplateTestSigrom NTTP arguments", "[vector][template][nttp]",
//   ((typename RangeQueryStruct, int X), RangeQueryStruct, X), (PrefixSums<double>, 0),
//   (SegmentTree<double>, 0)) {

// }

TEMPLATE_TEST_CASE("Interpolated<T> queries with different underlying data structures", "",
                   PrefixSums<DefaultGroup<double>>, SegmentTree<DefaultGroup<double>>) {
    using TestedType = Interpolated<TestType>;
    SECTION("Can be created on empty data") {
        std::vector<double> data;
        TestedType t{data.begin(), data.end()};
        REQUIRE(t.SoftQuery(0, 0) == 0);
        REQUIRE(t.SoftQuery(0, .1) == 0);
        REQUIRE(t.SoftQuery(-.1, 0) == 0);
        REQUIRE(t.SoftQuery(-5, 5) == 0);
    }

    SECTION("Correctly sums sequences of ones") {
        std::vector<double> data{101, 1.};
        TestedType t{data.begin(), data.end()};
        REQUIRE(t.SoftQuery(0, 1.) == Approx(1.));
        REQUIRE(t.SoftQuery(0, .1) == Approx(.1));
        REQUIRE(t.SoftQuery(.9, 1.1) == Approx(.2));
        REQUIRE(t.SoftQuery(.9, 2.1) == Approx(2.2));
        REQUIRE(t.SoftQuery(.9, 50) == Approx(49.1));
        REQUIRE(t.SoftQuery(.9, 50.9) == Approx(50));
        REQUIRE(t.SoftQuery(1, 50.9) == Approx(49.9));
    }

}