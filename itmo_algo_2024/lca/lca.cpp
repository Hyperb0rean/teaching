
#include <algorithm>
#include <iostream>
#include <istream>
#include <iterator>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>

namespace solution {

namespace traverse {

template <class Vertex, class Graph, class Visitor>
auto DepthFirstSearch(Vertex origin_vertex, Graph& graph, Visitor visitor,
                      std::unordered_set<Vertex>& visited) -> void {
    visitor.DiscoverVertex(origin_vertex);
    visited.insert(origin_vertex);
    for (auto& edge : graph.GetOutgoingEdges(origin_vertex)) {
        visitor.ExamineEdge(edge);
        if (const auto& target = graph.GetTarget(edge); !visited.contains(target)) {
            DepthFirstSearch(target, graph, visitor, visited);
            visitor.DropFromChild(origin_vertex);
        }
    }
    visitor.Drop();
}

template <class Vertex, class Edge>
class GraphVisitor {
public:
    virtual auto DiscoverVertex(Vertex /*vertex*/) -> void = 0;
    virtual auto ExamineEdge(Edge& /*edge*/) -> void = 0;
    virtual auto DropFromChild(Vertex /*vertex*/) -> void = 0;
    virtual auto Drop() -> void = 0;
    virtual ~GraphVisitor() = default;
};
}  // namespace traverse

namespace tree {

using Vertex = int;

struct Edge {
    Vertex from;
    Vertex to;
};

class Tree {
public:
    auto GetTarget(const Edge& edge) const noexcept -> Vertex {
        return edge.to;
    }

    auto GetOutgoingEdges(Vertex node) {
        return std::ranges::ref_view(adj_list_[node]);
    }

    auto GetOutgoingEdges(Vertex node) const {
        return std::ranges::ref_view(adj_list_[node]);
    }

    auto AddEdge(Vertex from, Vertex to) -> void {
        adj_list_[from].push_back({from, to});
        adj_list_[to].push_back({to, from});
        ++edges_;
    }

    auto Root() const -> Vertex {
        return 0;
    };

    auto Size() const -> int {
        return adj_list_.size();
    }

    Tree(int size) {
        adj_list_.resize(size);
    }

private:
    std::vector<std::vector<Edge>> adj_list_;
    int edges_;
};

}  // namespace tree

namespace io {

struct Query {
    int left_id, right_id, charge;
};

struct Test {
    std::vector<tree::Edge> tree;
    std::vector<Query> queries;
};

auto ReadInput(std::istream& is = std::cin) -> Test {
    Test test{};
    int tree_size;
    is >> tree_size;
    test.tree.resize(tree_size - 1);
    for (auto& edge : test.tree) {
        is >> edge.from >> edge.to;
    }
    int query_size;
    is >> query_size;
    test.queries.resize(query_size);
    for (auto& query : test.queries) {
        is >> query.left_id >> query.right_id >> query.charge;
    }
    return test;
}

auto PrintOutput(std::vector<int>&& output, std::ostream& os = std::cout) -> void {
    for (auto&& val : output) {
        os << ((val == 0) ? "No" : "Yes") << "\n";
    }
}

}  // namespace io

auto MakeTree(std::vector<tree::Edge>&& tree) -> tree::Tree {
    tree::Tree result(tree.size() + 1);
    for (auto&& [from, to] : tree) {
        result.AddEdge(from, to);
    }
    return result;
}

struct Index {
    std::vector<int> depth_;
    std::vector<tree::Vertex> vertex_;
    std::vector<int> vertex_to_depth_index_;
};

auto MakeIndex(tree::Tree&& tr) -> Index {
    using namespace solution::tree;
    Index result;
    result.vertex_to_depth_index_.resize(tr.Size());
    struct IndexVisitor : traverse::GraphVisitor<Vertex, Edge> {
        auto DiscoverVertex(Vertex vertex) -> void override {
            index_.vertex_to_depth_index_[vertex] = index_.depth_.size();
            index_.depth_.push_back(++depth);
            index_.vertex_.push_back(vertex);
        }
        virtual auto ExamineEdge(Edge& /*edge*/) -> void override {
        }
        virtual auto Drop() -> void override {
            --depth;
        }
        virtual auto DropFromChild(Vertex vertex) -> void override {
            index_.depth_.push_back(depth);
            index_.vertex_.push_back(vertex);
        }

        IndexVisitor(Index& index) : index_(index) {
        }

        Index& index_;
        int depth = 0;
    } visitor{result};

    std::unordered_set<Vertex> visited;
    solution::traverse::DepthFirstSearch(tr.Root(), tr, visitor, visited);

    return result;
}

class RMQ {
public:
    RMQ(std::vector<int>&& d) : depth_(std::move(d)) {
        PrecalcFloor();
        PrecalcSparseTable();
    }

    auto Query(int left, int right) const -> int {
        int j = floor_[right - left + 1];
        int const idx1 = sparse_table_[j][left];
        int const idx2 = sparse_table_[j][right - (1 << j) + 1];
        return (depth_[idx1] < depth_[idx2]) ? idx1 : idx2;
    }

    auto Distance(int left, int right) const -> int {
        int root = Query(left, right);
        return (depth_[left] - depth_[root]) + (depth_[right] - depth_[root]);
    }

private:
    auto PrecalcSparseTable() -> void {
        int const log = std::log2(std::ssize(depth_)) + 1;
        sparse_table_.resize(log, std::vector<int>(std::ssize(depth_), 0));
        for (int i : std::views::iota(0, std::ssize(depth_))) {
            sparse_table_[0][i] = i;
        }
        for (int j = 1; (1 << j) <= std::ssize(depth_); ++j) {
            for (int i = 0; i <= std::ssize(depth_) - (1 << j); ++i) {
                int const idx1 = sparse_table_[j - 1][i];
                int const idx2 = sparse_table_[j - 1][i + (1 << (j - 1))];
                sparse_table_[j][i] = (depth_[idx1] < depth_[idx2]) ? idx1 : idx2;
            }
        }
    }

    auto PrecalcFloor() -> void {
        floor_.resize(depth_.size() + 1, 0);
        for (int i : std::views::iota(2, std::ssize(depth_) + 1)) {
            floor_[i] = floor_[i >> 1] + 1;
        }
    }

    std::vector<int> depth_;
    std::vector<std::vector<int>> sparse_table_;
    std::vector<int> floor_;
};

class LCA {
public:
    LCA(Index&& index)
        : rmq_(std::move(index.depth_)),
          vertex_(std::move(index.vertex_)),
          vertex_to_depth_index_(std::move(index.vertex_to_depth_index_)) {
    }

    auto Query(tree::Vertex first, tree::Vertex second) -> tree::Vertex {
        auto&& [left, right] =
            std::minmax(vertex_to_depth_index_[first], vertex_to_depth_index_[second]);
        return vertex_[rmq_.Query(left, right)];
    }

    auto Distance(tree::Vertex first, tree::Vertex second) -> tree::Vertex {
        auto&& [left, right] =
            std::minmax(vertex_to_depth_index_[first], vertex_to_depth_index_[second]);
        return rmq_.Distance(left, right);
    }

private:
    RMQ rmq_;
    std::vector<tree::Vertex> vertex_;
    std::vector<int> vertex_to_depth_index_;
};

auto CheckConnectivity(Index&& index, std::vector<io::Query>&& queries) -> std::vector<int> {
    LCA lca{std::move(index)};

    std::vector output(queries.size(), 0);
    for (int i : std::views::iota(0, std::ssize(queries))) {
        output[i] =
            (lca.Distance(queries[i].left_id, queries[i].right_id) > queries[i].charge) ? 0 : 1;
    }
    return output;
}

}  // namespace solution

int main() {
    using namespace solution;

    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::ios::sync_with_stdio(false);

    auto&& test = io::ReadInput();
    auto&& index = MakeIndex(MakeTree(std::move(test.tree)));
    io::PrintOutput(CheckConnectivity(std::move(index), std::move(test.queries)));
}