#include "testlib.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <istream>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <set>

namespace lib {

namespace algo {

template <class Predicate>
auto BinarySearch(int left, int right, Predicate predicate = Predicate{}) -> int {
    while (right - left > 1) {
        auto &&mid = std::midpoint(left, right);
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
    auto &&index = BinarySearch(-1, static_cast<int>(std::distance(left, right)),
                                [predicate, &left](int mid) { return predicate(*(left + mid)); });
    return left + index;
}

}  // namespace algo

namespace traverse {
template <class Vertex, class Graph, class Visitor>
auto BreadthFirstSearch(Vertex origin_vertex, Graph *graph, Visitor visitor) -> void {
    std::unordered_set<Vertex> visited;
    std::queue<Vertex> current;
    visited.insert(origin_vertex);
    current.push(origin_vertex);
    visitor.DiscoverVertex(origin_vertex);

    while (!current.empty() && visitor.ShouldRun()) {
        auto &&vertex = current.front();
        visitor.ExamineVertex(vertex);
        current.pop();

        for (auto &&edge : graph->GetOutgoingEdges(vertex)) {
            visitor.ExamineEdgeConst(edge);
            visitor.ExamineEdge(&edge);
            auto &&target = graph->GetTarget(edge);
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
    virtual auto ExamineEdgeConst(const Edge & /*edge*/) -> void {
    }
    virtual auto ExamineEdge(Edge * /*edge*/) -> void {
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
template <class Vertex, class Edge>
class Graph {
public:
    using Vertex_t = Vertex;
    using Edge_t = Edge;

    auto EdgesCount() const noexcept -> int {
        return static_cast<int>(edges_.size());
    }

    auto GetDegree(Vertex node) const noexcept -> int {
        return (NodesCount() - node - 1 > 0 ? starts_[node + 1] : EdgesCount()) - starts_[node];
    }

    auto NodesCount() const noexcept -> int {
        return static_cast<int>(starts_.size());
    }

public:
    auto GetTarget(const Edge &edge) const noexcept -> Vertex {
        return edge.to;
    }

    auto GetOutgoingEdges(Vertex node) {
        return std::ranges::ref_view(edges_) | std::views::drop(starts_[node]) |
               std::views::take(GetDegree(node));
    }

    auto GetOutgoingEdges(Vertex node) const {
        return std::ranges::ref_view(edges_) | std::views::drop(starts_[node]) |
               std::views::take(GetDegree(node));
    }

protected:
    explicit Graph(int nodes_count, int edges_count) : starts_(nodes_count), edges_(edges_count) {
    }

    std::vector<Vertex> starts_;
    std::vector<Edge> edges_;

    // Edge array must be sorted
    auto FindEdge(Vertex from, Vertex to) -> std::vector<Edge>::iterator {
        auto &&edges = GetOutgoingEdges(from);
        auto iter = algo::BinarySearch(edges.begin(), edges.end(),
                                       [&to](auto &&mid) { return mid.to >= to; });
        return iter->to == to ? iter : edges_.end();
    }
};

}  // namespace graph

namespace flow {
template <class FlowNetwork>
class FlowNetworkBuilder;

using Vertex = int;

template <class Weight, class Vertex = ::lib::flow::Vertex>
struct Edge {
    using Weight_t = Weight;
    Vertex from;
    Vertex to;
    Weight weight;
    Edge *back_edge;
};

template <class Weight>
class FlowNetwork final : public graph::Graph<Vertex, Edge<Weight>> {
public:
    Vertex GetSource() const noexcept {
        return source_;
    }

    Vertex GetSink() const noexcept {
        return sink_;
    }

    friend FlowNetworkBuilder<FlowNetwork<Weight>>;

private:
    Vertex source_;
    Vertex sink_;

    explicit FlowNetwork(Vertex source, Vertex sink, int nodes_count, int edges_count)
        : graph::Graph<Vertex, Edge<Weight>>(nodes_count, edges_count),
          source_(source),
          sink_(sink) {
    }
};

template <class FlowNetwork, class Predicate>
class FilteredFlowNetwork {
public:
    FilteredFlowNetwork(FlowNetwork *graph, Predicate function)
        : graph_(graph), function_(function) {
    }

    auto EdgesCount() const noexcept -> int {
        return graph_->EdgesCount();
    }

    auto GetDegree(FlowNetwork::Vertex_t node) const noexcept -> int {
        return graph_->GetDegree(node);
    }

    auto NodesCount() const noexcept -> int {
        return graph_->NodesCount();
    }

    auto GetSource() const noexcept -> FlowNetwork::Vertex_t {
        return graph_->GetSource();
    }

    auto GetSink() const noexcept -> FlowNetwork::Vertex_t {
        return graph_->GetSink();
    }

public:
    auto GetTarget(const FlowNetwork::Edge_t &edge) const noexcept -> FlowNetwork::Vertex_t {
        return graph_->GetTarget(edge);
    }

    auto GetOutgoingEdges(FlowNetwork::Vertex_t node) {
        return graph_->GetOutgoingEdges(node) | std::views::filter(function_);
    }

    auto GetOutgoingEdges(FlowNetwork::Vertex_t node) const {
        return graph_->GetOutgoingEdges(node) | std::views::filter(function_);
    }

private:
    FlowNetwork *graph_;
    Predicate function_;
};

template <class FlowNetwork>
class FlowNetworkBuilder {
private:
    using Edge = FlowNetwork::Edge_t;
    using Vertex = FlowNetwork::Vertex_t;
    using Weight = Edge::Weight_t;

public:
    auto AddEdge(Vertex from, Vertex to, Weight weight) -> decltype(*this) {
        adjacency_list_[from].emplace_back(from, to, weight, nullptr);
        ++edges_count_;
        adjacency_list_[to].emplace_back(to, from, Weight{}, nullptr);
        ++edges_count_;
        return *this;
    }

    auto SetNodes(int nodes_count) -> decltype(*this) {
        adjacency_list_.resize(nodes_count);
        return *this;
    }

    auto SetSource(Vertex source) -> decltype(*this) {
        source_ = source;
        return *this;
    }
    auto SetSink(Vertex sink) -> decltype(*this) {
        sink_ = sink;
        return *this;
    }

    auto Build() -> FlowNetwork {
        FlowNetwork graph{source_, sink_, static_cast<int>(adjacency_list_.size()), edges_count_};
        auto *graph_ptr = &graph;
        BuildStarts(graph_ptr);
        BuildEdges(graph_ptr);
        BuildBackEdges(graph_ptr);
        return graph;
    }

private:
    int edges_count_ = 0;
    Vertex source_, sink_;
    std::vector<std::vector<Edge>> adjacency_list_;

    class BackEdgesBuildingVisitor final : public traverse::BfsVisitor<Vertex, Edge> {
    public:
        BackEdgesBuildingVisitor(FlowNetwork *graph) : graph_(graph) {
        }

        auto ExamineEdge(Edge *edge) -> void override {
            if (edge->back_edge || edge->weight == Weight{}) {
                return;
            }
            auto iter = graph_->FindEdge(edge->to, edge->from);
            while (iter->weight != Weight{}) {
                ++iter;
            }
            edge->back_edge = &*iter;
            edge->back_edge->back_edge = edge;
        }

    private:
        FlowNetwork *graph_;
    };

    auto BuildStarts(FlowNetwork *graph) const -> void {
        if (adjacency_list_.empty()) {
            return;
        }
        graph->starts_[0] = 0;
        for (int i = 0; const auto &list : adjacency_list_) {
            if (++i < static_cast<int>(graph->starts_.size())) {
                graph->starts_[i] = graph->starts_[i - 1] + static_cast<int>(list.size());
            }
        }
    }

    auto BuildEdges(FlowNetwork *graph) -> void {
        for (auto edge_it = graph->edges_.begin(); auto &&list : adjacency_list_) {
            for (auto &&edge : list) {
                *edge_it++ = std::move(edge);
            }
        }
        std::sort(graph->edges_.begin(), graph->edges_.end(), [](const Edge &lhs, const Edge &rhs) {
            return std::tie(lhs.from, lhs.to) < std::tie(rhs.from, rhs.to);
        });
    }

    auto BuildBackEdges(FlowNetwork *graph) -> void {
        BackEdgesBuildingVisitor visitor{graph};
        traverse::BreadthFirstSearch(source_, graph, visitor);
    }
};
}  // namespace flow

namespace flow {
using traverse::BfsVisitor;

struct Weight {
    int flow_value;
    int capacity;
};

auto operator==(const Weight &lhs, const Weight &rhs) -> bool {
    return std::tie(lhs.capacity, lhs.flow_value) == std::tie(rhs.capacity, rhs.flow_value);
}

using Path = std::vector<Edge<Weight> *>;

template <class Graph>
class PathCollectingVisitor final : public BfsVisitor<Vertex, Edge<Weight>> {
public:
    explicit PathCollectingVisitor(Graph *graph, Path *path) : path_(path), graph_(graph) {
    }

    auto ExamineEdge(Edge<Weight> *edge) -> void {
        (*path_)[edge->to] = edge;
    }

    auto ShouldRun() -> bool {
        return (*path_)[graph_->GetSink()] == nullptr;
    }

private:
    Path *path_;
    Graph *graph_;
};

auto FindAuxilaryPath(FlowNetwork<Weight> *graph) -> std::optional<Path> {
    Path path(graph->NodesCount(), nullptr);

    FilteredFlowNetwork filtered_network{
        graph, [graph, &path](const auto &edge) {
            return path[edge.to] == nullptr && edge.to != graph->GetSource() &&
                   edge.weight.capacity - edge.weight.flow_value > 0;
        }};
    PathCollectingVisitor visitor{&filtered_network, &path};

    traverse::BreadthFirstSearch(graph->GetSource(), &filtered_network, visitor);
    if (path[graph->GetSink()]) {
        return path;
    }
    return std::nullopt;
}

auto EdmundsKarp(FlowNetwork<Weight> *graph, int infinity = 1e9) -> int {
    int flow = 0;
    while (auto path = FindAuxilaryPath(graph)) {
        int delta_flow = infinity;
        for (auto edge = (*path)[graph->GetSink()]; edge != nullptr; edge = (*path)[edge->from]) {
            delta_flow = std::min(delta_flow, edge->weight.capacity - edge->weight.flow_value);
        }
        for (auto edge = (*path)[graph->GetSink()]; edge != nullptr; edge = (*path)[edge->from]) {
            edge->weight.flow_value += delta_flow;
            edge->back_edge->weight.flow_value -= delta_flow;
        }
        flow += delta_flow;
    }
    return flow;
}

auto ClearFlow(FlowNetwork<Weight> *network) -> void {
    for (auto &&node : std::views::iota(0, network->NodesCount())) {
        for (auto &&edge : network->GetOutgoingEdges(node)) {
            edge.weight.flow_value = 0;
            edge.back_edge->weight.flow_value = 0;
        }
    }
}

struct Bipartition {
    std::set<int> set_s;
    std::set<int> set_t;
};

class ComponentCollectingVisitor final : public BfsVisitor<int, Edge<Weight>> {
public:
    ComponentCollectingVisitor(std::set<int> &part) : part_(part) {
    }

    auto DiscoverVertex(int vertex) -> void override {
        part_.insert(vertex);
    }

private:
    std::set<int> &part_;
};

auto MinimaCut(FlowNetwork<Weight> *graph) -> Bipartition {
    Bipartition cut{};
    // Probably should replace
    EdmundsKarp(graph);
    ComponentCollectingVisitor collecter{cut.set_s};
    FilteredFlowNetwork filtered_graph{
        graph, [](const auto &edge) { return edge.weight.capacity > edge.weight.flow_value; }};

    traverse::BreadthFirstSearch(filtered_graph.GetSource(), &filtered_graph, collecter);
    for (auto &&vertex : std::views::iota(0, graph->NodesCount())) {
        if (!cut.set_s.contains(vertex)) {
            cut.set_t.insert(vertex);
        }
    }
    ClearFlow(graph);
    return cut;
}

}  // namespace flow
}  // namespace lib

namespace check {
using Subgraph = std::set<int>;

struct Edge {
    int from, to;
};

struct Test {
    int nodes_count;
    std::vector<Edge> edges;
};

namespace solution {

using FlowNetwork = ::lib::flow::FlowNetwork<::lib::flow::Weight>;

auto MakeFlowNetwork(Test &&test) -> FlowNetwork {
    using lib::flow::FlowNetworkBuilder;

    FlowNetworkBuilder<FlowNetwork> builder;
    const int nodes_count = test.nodes_count;
    const int edges_count = static_cast<int>(test.edges.size());
    const int source = 0;
    const int sink = nodes_count + 1;

    const int scale_to_int = nodes_count * (nodes_count - 1);

    builder
        .SetNodes(nodes_count + 2)  //
        .SetSource(source)          //
        .SetSink(sink);

    for (auto i : std::views::iota(1, nodes_count + 1)) {
        builder
            .AddEdge(source, i, {0, edges_count * scale_to_int})  //
            .AddEdge(i, sink, {0, 1});
    }

    for (auto &&edge : test.edges) {
        builder
            .AddEdge(edge.from, edge.to, {0, scale_to_int})  //
            .AddEdge(edge.to, edge.from, {0, scale_to_int});
    }

    return builder.Build();
}

auto UpdateSinkCapacities(FlowNetwork *graph, int guess) -> void {
    const int nodes_count = graph->NodesCount() - 2;
    const int edges_count = graph->EdgesCount() / 4 - nodes_count;
    const int scale_to_int = nodes_count * (nodes_count - 1);

    for (auto &&edge : graph->GetOutgoingEdges(graph->GetSink())) {
        edge.back_edge->weight.capacity = edges_count * scale_to_int + 2 * guess -
                                          (graph->GetDegree(edge.to) / 2 - 1) * scale_to_int;
    }
}

auto FindMaxDensitySubgraph(FlowNetwork *graph) -> Subgraph {
    const int nodes_count = graph->NodesCount() - 2;
    const int edges_count = graph->EdgesCount() / 4 - nodes_count;
    const int scale_to_int = nodes_count * (nodes_count - 1);
    const int lower = 0;
    const int upper = edges_count * scale_to_int;

    std::set<int> subgraph;
    lib::algo::BinarySearch(lower, upper, [&graph, &subgraph, edges_count](int guess) {
        UpdateSinkCapacities(graph, guess);
        const auto mincut = lib::flow::MinimaCut(graph);
        if (mincut.set_s.size() == 1) {
            if (guess == edges_count) {
                subgraph = mincut.set_t;
                subgraph.erase(graph->GetSink());
            }
            return true;
        } else {
            subgraph = mincut.set_s;
            subgraph.erase(graph->GetSource());
            return false;
        }
    });

    return subgraph;
}

}  // namespace solution

namespace io {
auto ReadUserInput() -> Subgraph {
    // TODO: validation of right order
    auto &&subgraph_size = ouf.readInt();
    Subgraph result_subgraph;
    while (subgraph_size--) {
        if (!result_subgraph.insert(ouf.readInt()).second) {
            quitf(_pe, "Duplication of vertecies");
        }
    }
    return result_subgraph;
}

auto ReadTestInput() -> Test {
    const int nodes_count = inf.readInt();
    const int edges_count = inf.readInt();
    std::vector<Edge> edges(edges_count);
    for (auto &&edge : edges) {
        edge = {.from = inf.readInt(), .to = inf.readInt()};
    }
    return {nodes_count, std::move(edges)};
}
}  // namespace io

struct Verdict {
    TResult result;
    std::string message;
};

template <class Graph>
auto CountEdges(const Graph &graph, const Subgraph &subgraph) -> int {
    int result;
    for (auto &&node : std::views::iota(0, graph.NodesCount())) {
        for (auto &&edge : graph.GetOutgoingEdges(node)) {
            if (subgraph.contains(node) && subgraph.contains(graph.GetTarget(edge))) {
                ++result;
            }
        }
    }
    return result;
}

template <class Graph>
auto Check(const Graph &network, Subgraph &&user, Subgraph &&test) -> Verdict {
    if (user == test) {
        return {_ok, "Test passed!"};
    }

    if (user.size() > network.NodesCount() - 2) {
        return {_wa, "Wrong graph"};
    }

    const int test_edges = CountEdges(network, test);
    const int user_edges = CountEdges(network, user);

    if (user_edges * test.size() >= test_edges * user.size()) {
        return {_ok, "Test passed!"};
    }

    return {_wa, "Test failed!"};
}

}  // namespace check

auto main(int argc, char *argv[]) -> int {

    std::ios::sync_with_stdio(false);
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    setName("Max density subgraph problem");
    registerTestlibCmd(argc, argv);

    auto &&user_subgraph = check::io::ReadUserInput();
    auto &&flow_network = check::solution::MakeFlowNetwork(check::io::ReadTestInput());
    auto &&test_subgraph = check::solution::FindMaxDensitySubgraph(&flow_network);
    auto &&[result, message] =
        check::Check(flow_network, std::move(user_subgraph), std::move(test_subgraph));

    quitf(result, "%s", message.data());
}