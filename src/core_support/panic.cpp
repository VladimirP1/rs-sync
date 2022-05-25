#include "panic.hpp"

#include <fstream>
#include <string>
#include <cassert>

void panic_to_file(const char* reason, bool should_panic) {
    if (should_panic) {
        std::ofstream out("panic.txt");
        out << reason << std::endl;
        out.close();
        assert(!should_panic);
        exit(1);
    }
}