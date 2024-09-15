
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <istream>
#include <iterator>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <numeric>

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
        }
    }
    visitor.ExamineVertex(origin_vertex);
}

template <class Vertex, class Edge>
class GraphVisitor {
public:
    virtual auto DiscoverVertex(Vertex /*vertex*/) -> void = 0;
    virtual auto ExamineEdge(Edge& /*edge*/) -> void = 0;
    virtual auto ExamineVertex(Vertex /*vertex*/) -> void = 0;
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
    auto EdgesCount() const noexcept -> int {
        return edges_;
    }

    auto GetDegree(Vertex node) const noexcept -> int {
        return adj_list_.at(node).size();
    }

    auto NodesCount() const noexcept -> int {
        return adj_list_.size();
    }

    auto GetTarget(const Edge& edge) const noexcept -> Vertex {
        return edge.to;
    }

    auto GetOutgoingEdges(Vertex node) {
        return std::ranges::ref_view(adj_list_[node]);
    }

    auto GetOutgoingEdges(Vertex node) const {
        return std::ranges::ref_view(adj_list_.at(node));
    }

    auto AddEdge(Vertex from, Vertex to) -> void {
        adj_list_[from].push_back({from, to});
        adj_list_[to].push_back({to, from});
        ++edges_;
    }

    auto Root() const noexcept -> Vertex {
        return root_;
    }

    auto SetRoot(Vertex root) -> void {
        root_ = root;
    }

private:
    std::unordered_map<Vertex, std::vector<Edge>> adj_list_;
    int edges_;
    Vertex root_;
};

}  // namespace tree

namespace io {

struct Query {
    int left_id, right_id;
};

struct Test {
    tree::Vertex root;
    std::vector<tree::Edge> tree;
    std::vector<Query> queries;
};

auto ReadInput(std::istream& is = std::cin) -> Test {
    Test test{};
    int tree_size;
    is >> tree_size >> test.root;
    test.tree.resize(tree_size);
    for (auto& edge : test.tree) {
        is >> edge.from >> edge.to;
    }
    int query_size;
    is >> query_size;
    test.queries.resize(query_size);
    for (auto& query : test.queries) {
        is >> query.left_id >> query.right_id;
    }
    return test;
}

}  // namespace io
auto MakeTree(io::Test const& test) -> tree::Tree {
    tree::Tree result;
    result.SetRoot(test.root);
    for (auto&& [from, to] : test.tree) {
        result.AddEdge(from, to);
    }
    return result;
}

struct Index {
    std::vector<int> depth_;
    std::vector<tree::Vertex> vertex_;
    std::unordered_map<tree::Vertex, int> vertex_to_depth_index_;
};

auto MakeIndex(tree::Tree&& tr) -> Index {
    using namespace solution::tree;
    Index result;
    struct IndexVisitor : traverse::GraphVisitor<Vertex, Edge> {
        auto DiscoverVertex(Vertex vertex) -> void override {
            index_.vertex_to_depth_index_[vertex] = index_.depth_.size();
            index_.depth_.push_back(depth++);
            index_.vertex_.push_back(vertex);
        }
        virtual auto ExamineEdge(Edge& /*edge*/) -> void override {
            //
        }
        virtual auto ExamineVertex(Vertex /*vertex*/) -> void override {
            --depth;
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

namespace rmq {
auto RMQ(std::vector<int> const& index, int left, int right) -> int {
}
}  // namespace rmq

auto LCA() -> tree::Vertex {
}

}  // namespace solution

int main() {
    using namespace solution;

    auto&& test = io::ReadInput();
    auto&& index = MakeIndex(MakeTree(test));
    // struct PrintingVisitor : traverse::GraphVisitor<Vertex, Edge> {
    //     auto DiscoverVertex(Vertex vertex) -> void override {
    //         std::cout << "Discover: " << vertex << "\n";
    //     }
    //     virtual auto ExamineEdge(Edge& edge) -> void override {
    //         std::cout << "Examine edge: " << edge.from << " " << edge.to << "\n";
    //     }
    //     virtual auto ExamineVertex(Vertex vertex) -> void override {
    //         std::cout << "Examine: " << vertex << "\n";
    //     }
    // } visitor;

    // std::unordered_set<Vertex> visited;
    // traverse::DepthFirstSearch(test.root, tree, visitor, visited);

    // for (auto&& d : index.depth_) {
    //     std::cout << d << ' ';
    // }
    // std::cout << "\n";
    // for (auto&& v : index.vertex_) {
    //     std::cout << v << ' ';
    // }
    // std::cout << "\n";
    // for (auto&& [v, d] : index.vertex_to_depth_index_) {
    //     std::cout << v << ' ' << d << "  ";
    // }
}