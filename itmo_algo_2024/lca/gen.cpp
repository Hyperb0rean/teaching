#include "testlib.h"

#include <iostream>
#include <utility>
#include <vector>
#include <set>
#include <limits>

namespace gen {
constexpr int kMaxNumber = 10'000;
constexpr int kMaxQueries = 1'000'000;
constexpr int kMaxId = kMaxNumber;
constexpr int kMaxCharge = kMaxNumber;

auto GenerateIds(int N) -> std::vector<int> {
    // std::set<int> unique_ids;
    // while (unique_ids.size() < N) {
    //     unique_ids.insert(rnd.next(1, kMaxId));
    // }
    // return std::vector<int>(unique_ids.begin(), unique_ids.end());

    std::vector<int> ids(N, 0);

    for (int i = 1; i != N; ++i) {
        ids[i] = i;
    }
    return ids;
}

auto GenerateTree(std::vector<int> const& ids) -> void {
    int const N = ids.size();
    int const t = rnd.next(1, N - 1);
    std::vector<int> p(N);

    for (int i = 1; i < N; i++) {
        p[i] = rnd.wnext(i, t);
    }

    std::vector<std::pair<int, int>> edges;
    for (int i = 1; i < N; i++) {
        if (rnd.next(2)) {
            edges.push_back(std::make_pair(ids[i], ids[p[i]]));
        } else {
            edges.push_back(std::make_pair(ids[p[i]], ids[i]));
        }
    }
    shuffle(edges.begin(), edges.end());

    std::cout << N << std::endl;
    for (const auto& edge : edges) {
        std::cout << edge.first << " " << edge.second << std::endl;
    }
}

auto GenerateQueries(int Q, std::vector<int> const& ids) -> void {
    int const N = ids.size();
    for (int i = 0; i < Q; ++i) {
        int u = ids[rnd.next(0, N - 1)];
        int v = ids[rnd.next(0, N - 1)];
        int T = rnd.next(0, kMaxCharge);
        std::cout << u << " " << v << " " << T << std::endl;
    }
}

}  // namespace gen

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);

    int N = rnd.next(2, gen::kMaxNumber);

    auto&& ids = gen::GenerateIds(N);

    gen::GenerateTree(ids);

    int Q = rnd.next(1, gen::kMaxQueries);
    std::cout << Q << std::endl;

    gen::GenerateQueries(Q, ids);

    return 0;
}