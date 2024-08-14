#include "../testlib.h"

#include <algorithm>
#include <iostream>
#include <istream>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lib {

namespace algo {

struct BasicStop {
    auto operator()(int left, int right) -> bool {
        return right - left > 1;
    }
};

template <class Predicate, class Stop = BasicStop>
auto BinarySearch(int left, int right, Predicate predicate, Stop stop = Stop{}) -> int {
    while (stop(left, right)) {
        auto mid = std::midpoint(left, right);
        if (predicate(mid)) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return right;
}

template <class Iterator, class Predicate>
auto BinarySearch(Iterator left, Iterator right, Predicate predicate = Predicate{})
    -> Iterator requires std::random_access_iterator<Iterator> {
    int index = BinarySearch(-1, std::distance(left, right),
                             [predicate, &left](int mid) { return predicate(*(left + mid)); });
    return left + index;
}

}  // namespace algo

namespace traverse {
template <class Vertex, class Graph, class Visitor>
auto BreadthFirstSearch(Vertex origin_vertex, Graph& graph, Visitor visitor) -> void {
    std::unordered_set<Vertex> visited;
    std::queue<Vertex> current;
    visited.insert(origin_vertex);
    current.push(origin_vertex);
    visitor.DiscoverVertex(origin_vertex);

    while (!current.empty() && visitor.ShouldRun()) {
        auto vertex = current.front();
        visitor.ExamineVertex(vertex);
        current.pop();

        for (auto& edge : graph.GetOutgoingEdges(vertex)) {
            visitor.ExamineEdge(edge);
            auto target = graph.GetTarget(edge);
            if (!visited.contains(target)) {
                visited.insert(target);
                current.push(target);
                visitor.DiscoverVertex(target);
            }
        }
    }
}

template <class Vertex, class Edge>
class BfsVisitor {
public:
    virtual auto DiscoverVertex(Vertex /*vertex*/) -> void {
    }
    virtual auto ExamineEdge(Edge& /*edge*/) -> void {
    }
    virtual auto ExamineVertex(Vertex /*vertex*/) -> void {
    }
    virtual auto ShouldRun() -> bool {
        return true;
    }
    virtual ~BfsVisitor() = default;
};
}  // namespace traverse

namespace graph {

template <class Weight>
class FlowNetworkBuilder;

template <class Weight>
struct Edge {
    int from;
    int to;
    Weight weight;
    Edge* back_edge;
};

template <class Weight>
class FlowNetwork {
    int source_;
    int sink_;

    std::vector<int> starts_;
    std::vector<Edge<Weight>> edges_;

    // Edge array must be sorted
    auto FindEdge(int from, int to) -> std::vector<Edge<Weight>>::iterator {
        auto edges = GetOutgoingEdges(from);
        auto iter = algo::BinarySearch(edges.begin(), edges.end(),
                                       [&to](auto mid) { return mid.to >= to; });
        return iter->to == to ? iter : edges_.end();
    }

    FlowNetwork(int source, int sink, int nodes_count, int edges_count)
        : source_(source), sink_(sink), starts_(nodes_count), edges_(edges_count) {
    }

public:
    auto EdgesCount() const noexcept -> int {
        return static_cast<int>(edges_.size());
    }

    auto GetDegree(int node) const noexcept -> int {
        return (node + 1 < NodesCount() ? starts_[node + 1] : EdgesCount()) - starts_[node];
    }

    auto NodesCount() const noexcept -> int {
        return static_cast<int>(starts_.size());
    }

    auto GetSource() const noexcept -> int {
        return source_;
    }

    auto GetSink() const noexcept -> int {
        return sink_;
    }

    auto GetTarget(const Edge<Weight>& edge) const noexcept -> int {
        return edge.to;
    }

    auto GetOutgoingEdges(int node) {
        return std::ranges::ref_view(edges_) | std::views::drop(starts_[node]) |
               std::views::take(GetDegree(node));
    }

    auto GetOutgoingEdges(int node) const {
        return std::ranges::ref_view(edges_) | std::views::drop(starts_[node]) |
               std::views::take(GetDegree(node));
    }

    friend FlowNetworkBuilder<Weight>;
};

template <class Weight, class Predicate>
class FilteredFlowNetwork {
    FlowNetwork<Weight>& graph_;
    Predicate function_;

public:
    FilteredFlowNetwork(FlowNetwork<Weight>& graph, Predicate function)
        : graph_(graph), function_(function) {
    }

    auto EdgesCount() const noexcept -> int {
        return graph_.EdgesCount();
    }

    auto GetDegree(int node) const noexcept -> int {
        return graph_.GetDegree(node);
    }

    auto NodesCount() const noexcept -> int {
        return graph_.NodesCount();
    }

    auto GetSource() const noexcept -> int {
        return graph_.GetSource();
    }

    auto GetSink() const noexcept -> int {
        return graph_.GetSink();
    }

    auto GetTarget(const Edge<Weight>& edge) const noexcept -> int {
        return graph_.GetTarget(edge);
    }

    auto GetOutgoingEdges(int node) {
        return graph_.GetOutgoingEdges(node) | std::views::filter(function_);
    }

    auto GetOutgoingEdges(int node) const {
        return graph_.GetOutgoingEdges(node) | std::views::filter(function_);
    }
};

template <class Weight>
class FlowNetworkBuilder {
    int edges_count_ = 0;
    int source_, sink_;
    std::vector<std::vector<Edge<Weight>>> adjacency_list_;

    class BackEdgesBuildingVisitor : public traverse::BfsVisitor<int, Edge<Weight>> {
        FlowNetwork<Weight>& graph_;

    public:
        BackEdgesBuildingVisitor(FlowNetwork<Weight>& graph) : graph_(graph) {
        }

        auto ExamineEdge(Edge<Weight>& edge) -> void override {
            if (!edge.back_edge) {
                auto iter = graph_.FindEdge(edge.to, edge.from);
                while (iter->weight != Weight{}) {
                    ++iter;
                }
                edge.back_edge = &*iter;
                edge.back_edge->back_edge = &edge;
            }
        }
    };

    auto BuildStarts(FlowNetwork<Weight>& graph) const -> void {
        if (!adjacency_list_.empty()) {
            graph.starts_[0] = 0;
            for (int i = 0; const auto& list : adjacency_list_) {
                if (++i < static_cast<int>(graph.starts_.size())) {
                    graph.starts_[i] = graph.starts_[i - 1] + list.size();
                }
            }
        }
    }

    auto BuildEdges(FlowNetwork<Weight>& graph) -> void {
        for (auto edge_it = graph.edges_.begin(); auto& list : adjacency_list_) {
            for (auto&& edge : list) {
                *edge_it++ = std::move(edge);
            }
        }
        std::sort(graph.edges_.begin(), graph.edges_.end(),
                  [](const Edge<Weight>& lhs, const Edge<Weight>& rhs) {
                      return std::tie(lhs.from, lhs.to) < std::tie(rhs.from, rhs.to);
                  });
    }

    auto BuildBackEdges(FlowNetwork<Weight>& graph) -> void {
        BackEdgesBuildingVisitor visitor{graph};
        traverse::BreadthFirstSearch(source_, graph, visitor);
    }

public:
    auto AddEdge(int from, int to, Weight weight) -> void {
        adjacency_list_[from].emplace_back(from, to, weight, nullptr);
        ++edges_count_;
        adjacency_list_[to].emplace_back(to, from, Weight{}, nullptr);
        ++edges_count_;
    }

    auto SetNodes(int nodes_count) -> void {
        adjacency_list_.resize(nodes_count);
    }

    auto SetSource(int source) -> void {
        source_ = source;
    }
    auto SetSink(int sink) -> void {
        sink_ = sink;
    }

    auto Build() -> FlowNetwork<Weight> {
        FlowNetwork<Weight> graph{source_, sink_, static_cast<int>(adjacency_list_.size()),
                                  edges_count_};
        BuildStarts(graph);
        BuildEdges(graph);
        BuildBackEdges(graph);
        return graph;
    }
};
}  // namespace graph

namespace flow {
using graph::Edge;
using graph::FilteredFlowNetwork;
using graph::FlowNetwork;
using traverse::BfsVisitor;

constexpr int kInfinity = 1000000;

struct Weight {
    int flow_value;
    int capacity;

    friend auto operator==(const Weight& lhs, const Weight& rhs) -> bool {
        return std::tie(lhs.capacity, lhs.flow_value) == std::tie(rhs.capacity, rhs.flow_value);
    }
};
using Path = std::vector<Edge<Weight>*>;

template <class Predicate>
class PathCollectingVisitor : public BfsVisitor<int, Edge<Weight>> {
    Path& path_;
    FilteredFlowNetwork<Weight, Predicate>& graph_;

public:
    explicit PathCollectingVisitor(FilteredFlowNetwork<Weight, Predicate>& graph, Path& path)
        : path_(path), graph_(graph) {
    }

    auto ExamineEdge(Edge<Weight>& edge) -> void {
        path_[edge.to] = &edge;
    }

    auto ShouldRun() -> bool {
        return path_[graph_.GetSink()] == nullptr;
    }
};

auto FindAuxilaryPath(FlowNetwork<Weight>& graph) -> std::optional<Path> {
    Path path(graph.NodesCount(), nullptr);

    auto filtered_network = FilteredFlowNetwork(graph, [&graph, &path](auto edge) {
        return path[edge.to] == nullptr && edge.to != graph.GetSource() &&
               edge.weight.capacity > edge.weight.flow_value;
    });
    PathCollectingVisitor visitor{filtered_network, path};

    traverse::BreadthFirstSearch(graph.GetSource(), filtered_network, visitor);
    if (path[graph.GetSink()]) {
        return path;
    }
    return std::nullopt;
}

auto EdmundsKarp(FlowNetwork<Weight>& graph) -> int {
    int flow = 0;
    while (auto path = FindAuxilaryPath(graph)) {
        int delta_flow = kInfinity;
        for (auto edge = (*path)[graph.GetSink()]; edge != nullptr; edge = (*path)[edge->from]) {
            delta_flow = std::min(delta_flow, edge->weight.capacity - edge->weight.flow_value);
        }
        for (auto edge = (*path)[graph.GetSink()]; edge != nullptr; edge = (*path)[edge->from]) {
            edge->weight.flow_value += delta_flow;
            edge->back_edge->weight.flow_value -= delta_flow;
        }
        flow += delta_flow;
    }
    return flow;
}

struct Bipartition {
    std::set<int> set_s;
    std::set<int> set_t;
};

class ComponentCollectingVisitor : public BfsVisitor<int, Edge<Weight>> {
public:
    ComponentCollectingVisitor(std::set<int>& part) : part_(part) {
    }

    auto DiscoverVertex(int vertex) -> void override {
        part_.insert(vertex);
    }

private:
    std::set<int>& part_;
};

auto ClearFlow(FlowNetwork<Weight>& network) -> void {
    for (auto node : std::views::iota(0, network.NodesCount())) {
        for (auto& edge : network.GetOutgoingEdges(node)) {
            edge.weight.flow_value = 0;
            edge.back_edge->weight.flow_value = 0;
        }
    }
}

auto MinimaCut(FlowNetwork<Weight>& graph) -> Bipartition {
    Bipartition cut{};
    // Probably should replace
    EdmundsKarp(graph);
    ComponentCollectingVisitor collecter{cut.set_s};
    auto filtered_graph = FilteredFlowNetwork(
        graph, [](auto edge) { return edge.weight.capacity > edge.weight.flow_value; });

    traverse::BreadthFirstSearch(filtered_graph.GetSource(), filtered_graph, collecter);
    for (auto vertex : std::views::iota(0, graph.NodesCount())) {
        if (!cut.set_s.contains(vertex)) {
            cut.set_t.insert(vertex);
        }
    }
    ClearFlow(graph);
    return cut;
}

}  // namespace flow

}  // namespace lib

namespace checker {
using Subgraph = std::vector<int>;
auto ReadUserInput() -> Subgraph {
  auto &&subgraph_size = ouf.readInt();
  Subgraph result(subgraph_size);
  for (auto &&vertex : result) {
    vertex = ouf.readInt();
  }
  return result;
}

auto ReadTestInput(std::istream& is = std::cin) -> lib::graph::FlowNetwork<lib::flow::Weight> {
    lib::graph::FlowNetworkBuilder<lib::flow::Weight> builder;
    int nodes_count = inf.readInt();
    int edges_count = inf.readInt();
    const int source = 0;
    const int sink = nodes_count + 1;

    const int scale_to_int = nodes_count * (nodes_count - 1);

    builder.SetNodes(nodes_count + 2);
    builder.SetSource(source);
    builder.SetSink(sink);

    for (auto i : std::views::iota(1, nodes_count + 1)) {
        builder.AddEdge(source, i, {0, edges_count * scale_to_int});
        builder.AddEdge(i, sink, {0, 1});
    }

    while (edges_count--) {
        int from = inf.readInt();
        int to = inf.readInt();
        builder.AddEdge(from, to, {0, scale_to_int});
        builder.AddEdge(to, from, {0, scale_to_int});
    }

    return builder.Build();
}
} // namespace checker

auto main(int argc, char *argv[]) -> int {

  std::ios::sync_with_stdio(false);
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.tie(nullptr);

  setName("Max density subgraph problem");
  registerTestlibCmd(argc, argv);

  auto &&subgraph = checker::ReadUserInput();
}
