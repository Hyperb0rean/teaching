#include "testlib.h"
#include <iostream>
#include <set>
#include <utility>

using Edges = std::set<std::pair<int, int>>;

auto GenerateEdges(int nodes_count) -> Edges {
    Edges result;
    const int edges_count = 1000;
    while (static_cast<int>(result.size()) < edges_count) {
        const int first = rnd.next(1, nodes_count);
        const int second = rnd.next(1, nodes_count);
        if (first == second) {
            continue;
        }
        if (result.contains(std::make_pair(first, second)) ||
            result.contains(std::make_pair(second, first))) {
            continue;
        }
        result.emplace(first, second);
    }
    return result;
}

auto main(int argc, char* argv[]) -> int {
    registerGen(argc, argv, 1);
    const int nodes_count = 100;
    auto&& edges = GenerateEdges(nodes_count);

    std::cout << nodes_count << " " << edges.size() << std::endl;

    for (auto&& edge : edges) {
        std::cout << edge.first << " " << edge.second << std::endl;
    }
}