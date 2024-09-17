#include <cstddef>
#include <vector>
#include <iostream>

#include "testlib.h"

namespace check {
namespace io {
auto ReadUserInput(size_t max_size) -> std::vector<std::string> {
    std::vector<std::string> result;
    result.reserve(max_size);
    while (max_size--) {
        if (ouf.eof()) {
            quitf(_pe, "Too few answers");
        }
        result.emplace_back(ouf.readString());
    }
    ouf.readString();
    if (!ouf.eof()) {
        quitf(_pe, "Too many answers");
    }
    return result;
}

auto ReadAnswer() -> std::vector<std::string> {
    std::vector<std::string> result;
    do {
        result.emplace_back(ans.readString());
    } while (!ans.eof());
    return result;
}
}  // namespace io
}  // namespace check

auto main(int argc, char* argv[]) -> int {

    // std::ios::sync_with_stdio(false);
    // std::ios_base::sync_with_stdio(false);
    // std::cin.tie(nullptr);
    // std::cout.tie(nullptr);

    setName("LCA min distance");
    registerTestlibCmd(argc, argv);

    auto&& answer = check::io::ReadAnswer();
    auto&& user = check::io::ReadUserInput(answer.size());

    for (int i = 0; i != std::ssize(answer); ++i) {
        if (answer[i] != user[i]) {
            quitf(_wa, "Its over! %d is not %s", i, user[i]);
        }
    }
    quitf(_ok, "Absolutely right!");
}