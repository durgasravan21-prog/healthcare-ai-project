"""Classical search demonstrations on a non-clinical workflow graph."""
from collections import deque
import heapq
import time

GRAPH = {
    "Start": ["Registration", "Emergency Desk"],
    "Registration": ["Basic Review"],
    "Emergency Desk": ["Basic Review", "Priority Review"],
    "Basic Review": ["Specialist Review", "Information Desk"],
    "Priority Review": ["Specialist Review"],
    "Information Desk": ["Specialist Review"],
    "Specialist Review": ["Follow-up Planning"],
    "Follow-up Planning": ["End"],
    "End": [],
}
EDGE_COSTS = {(a, b): 1 for a, ns in GRAPH.items() for b in ns}
HEURISTIC = {"Start": 4, "Registration": 3, "Emergency Desk": 3,
             "Basic Review": 2, "Priority Review": 2, "Information Desk": 2,
             "Specialist Review": 1, "Follow-up Planning": 1, "End": 0}


def _result(path, explored, started, cost=None):
    return {"path": path, "explored": explored,
            "steps": len(path) - 1 if path else None, "cost": cost,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}


def bfs(graph, start, goal):
    started = time.perf_counter()
    queue, visited, explored = deque([(start, [start])]), {start}, []
    while queue:
        node, path = queue.popleft()
        explored.append(node)
        if node == goal:
            return _result(path, explored, started, len(path) - 1)
        for nxt in graph.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return _result(None, explored, started)


def dfs(graph, start, goal):
    started = time.perf_counter()
    stack, visited, explored = [(start, [start])], set(), []
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        explored.append(node)
        if node == goal:
            return _result(path, explored, started, len(path) - 1)
        for nxt in reversed(graph.get(node, [])):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))
    return _result(None, explored, started)


def greedy_best_first(graph, start, goal):
    started = time.perf_counter()
    counter, frontier, visited, explored = 0, [(HEURISTIC[start], 0, start, [start])], set(), []
    while frontier:
        _, _, node, path = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        explored.append(node)
        if node == goal:
            cost = sum(EDGE_COSTS.get((a, b), 1) for a, b in zip(path, path[1:]))
            return _result(path, explored, started, cost)
        for nxt in graph.get(node, []):
            if nxt not in visited:
                counter += 1
                heapq.heappush(frontier, (HEURISTIC.get(nxt, 0), counter, nxt, path + [nxt]))
    return _result(None, explored, started)


def astar(graph, start, goal):
    started = time.perf_counter()
    counter, frontier, best, explored = 0, [(HEURISTIC[start], 0, 0, start, [start])], {start: 0}, []
    while frontier:
        _, cost, _, node, path = heapq.heappop(frontier)
        if cost != best.get(node):
            continue
        explored.append(node)
        if node == goal:
            return _result(path, explored, started, cost)
        for nxt in graph.get(node, []):
            new_cost = cost + EDGE_COSTS.get((node, nxt), 1)
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                counter += 1
                heapq.heappush(frontier, (new_cost + HEURISTIC.get(nxt, 0),
                                          new_cost, counter, nxt, path + [nxt]))
    return _result(None, explored, started)


def hill_climbing(graph, start, goal):
    started, current = time.perf_counter(), start
    path, explored = [start], [start]
    while current != goal:
        neighbours = graph.get(current, [])
        if not neighbours:
            return _result(None, explored, started)
        candidate = min(neighbours, key=lambda n: HEURISTIC.get(n, float("inf")))
        if HEURISTIC.get(candidate, float("inf")) >= HEURISTIC.get(current, float("inf")):
            return _result(None, explored, started)
        current = candidate
        path.append(current)
        explored.append(current)
    cost = sum(EDGE_COSTS.get((a, b), 1) for a, b in zip(path, path[1:]))
    return _result(path, explored, started, cost)


def run_all(start="Start", goal="End"):
    if start not in GRAPH or goal not in GRAPH:
        raise ValueError("Choose valid start and goal nodes.")
    return {"BFS": bfs(GRAPH, start, goal), "DFS": dfs(GRAPH, start, goal),
            "Greedy Best-First": greedy_best_first(GRAPH, start, goal),
            "Hill Climbing": hill_climbing(GRAPH, start, goal),
            "A*": astar(GRAPH, start, goal)}
