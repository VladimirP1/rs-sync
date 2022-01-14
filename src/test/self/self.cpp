#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

SCENARIO("Smoke test of tests") {
    GIVEN("Noting") { REQUIRE(true); }
}