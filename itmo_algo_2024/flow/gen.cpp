#include "testlib.h"
#include <iostream>
#include <set>
#include <utility>

using Edges = std::set<std::pair<int, int>>;

auto GenerateEdges(int nodes_count) -> Edges {
    Edges result;
    const int max_possible = std::min(1000, nodes_count * (nodes_count - 1) / 2);
    const int edges_count = rnd.next(max_possible);
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
    const int nodes_count = rnd.next(1, 100);
    auto&& edges = GenerateEdges(nodes_count);

    std::cout << nodes_count << " " << edges.size() << std::endl;

    for (auto&& edge : edges) {
        std::cout << edge.first << " " << edge.second << std::endl;
    }
}