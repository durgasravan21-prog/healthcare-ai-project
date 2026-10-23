from pathlib import Path
import pandas as pd
from collections import deque
import heapq
# Illustrative healthcare pathway graph
HEALTHCARE_GRAPH = {
    "Assessment": ["Treatment_A", "Treatment_B"],
    "Treatment_A": ["Followup_A", "Followup_B"],
    "Treatment_B": ["Followup_B"],
    "Followup_A": ["Outcome"],
    "Followup_B": ["Outcome"],
    "Outcome": []
}


#A* search edge cost

EDGE_COSTS = {
    ("Assessment", "Treatment_A"): 2,
    ("Assessment", "Treatment_B"): 4,
    ("Treatment_A", "Followup_A"): 3,
    ("Treatment_A", "Followup_B"): 2,
    ("Treatment_B", "Followup_B"): 2,
    ("Followup_A", "Outcome"): 2,
    ("Followup_B", "Outcome"): 1
}

HEURISTIC = {
    "Assessment": 5,
    "Treatment_A": 3,
    "Treatment_B": 3,
    "Followup_A": 2,
    "Followup_B": 1,
    "Outcome": 0
}


# Breadth first search 
def bfs(graph, start, goal):
    # Check that the starting state exists
    if start not in graph:
        raise ValueError(f"Unknown start state: {start}")

    # Queue stores (current state, path to that state)
    queue = deque([(start, [start])])

    # Track states already discovered
    discovered = {start}

    # Record the order in which states are explored
    exploration_order = []

    while queue:
        current, path = queue.popleft()

        exploration_order.append(current)

        # Check whether we reached the goal
        if current == goal:
            return {
                "path": path,
                "exploration_order": exploration_order,
                "goal_found": True
            }

        # Explore the current state's neighbors
        for neighbor in graph.get(current, []):
            if neighbor not in discovered:
                discovered.add(neighbor)

                # Extend the path with the new neighbor
                new_path = path + [neighbor]

                queue.append((neighbor, new_path))

# Return this if the goal cannot be reached
    return {
        "path": [],
        "exploration_order": exploration_order,
        "goal_found": False
    }


def dfs(graph, start, goal):
    if start not in graph:
        raise ValueError(f"Unknown start state: {start}")

    # Stack stores the current state and its path
    stack = [(start, [start])]

    discovered = {start}
    exploration_order = []

    while stack:
        current, path = stack.pop()

        exploration_order.append(current)

        if current == goal:
            return {
                "path": path,
                "exploration_order": exploration_order,
                "goal_found": True
            }

        # Reverse neighbors to preserve the graph's listed order
        for neighbor in reversed(graph.get(current, [])):
            if neighbor not in discovered:
                discovered.add(neighbor)
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))

    return {
        "path": [],
        "exploration_order": exploration_order,
        "goal_found": False
    }



# A* search algorithm implementation
def a_star(graph, edge_costs, heuristic, start, goal):
    if start not in graph:
        raise ValueError(f"Unknown start state: {start}")

    if goal not in graph:
        raise ValueError(f"Unknown goal state: {goal}")

    priority_queue = [(heuristic[start], 0, start, [start])]
    best_cost = {start: 0}
    exploration_order = []
    
    #Process the priority queue
    while priority_queue:
        f_score, g_cost, current, path = heapq.heappop(priority_queue)

        if g_cost > best_cost.get(current, float("inf")):
            continue

        exploration_order.append(current)

        if current == goal:
            return {
                "path": path,
                "exploration_order": exploration_order,
                "goal_found": True,
                "path_cost": g_cost
            }
        
        for neighbor in graph.get(current, []):
            edge_cost = edge_costs[(current, neighbor)]
            new_g = g_cost + edge_cost

            if new_g < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_g
                new_f = new_g + heuristic[neighbor]
                new_path = path + [neighbor]

                heapq.heappush(
                    priority_queue,
                    (new_f, new_g, neighbor, new_path)
                )
    return {
        "path": [],
        "exploration_order": exploration_order,
        "goal_found": False,
        "path_cost": None
    }



def search_dataset_by_symptom(symptom):
    project_dir = Path(__file__).resolve().parent
    csv_file = (
        project_dir
        / "data"
        / "synthetic_healthcare_records_500.csv"
    )

    df = pd.read_csv(csv_file)

    matches = df[
        df["reported_symptoms"].str.contains(
            symptom, case=False, na=False, regex=False
        )
    ]

    
    return matches[
        [
            "patient_id",
            "age",
            "age_group",
            "reported_symptoms",
            "synthetic_condition_label",
            "category",
            "synthetic_treatment_category"
        ]
    ]


def filter_healthcare_records(
    symptom="",
    age_group="",
    category=""
):
    
    
    project_dir = Path(__file__).resolve().parent

    csv_file = (
        project_dir
        / "data"
        / "synthetic_healthcare_records_500.csv"
    )

    print("Dataset path:", csv_file)


    df = pd.read_csv(csv_file)

    if symptom:
        df = df[
            df["reported_symptoms"].str.contains(
                symptom,
                case=False,
                na=False,
                regex=False
            )
        ]

    if age_group:
        df = df[
            df["age_group"].str.casefold()
            == age_group.casefold()
        ]

    if category:
        df = df[
            df["category"].str.casefold()
            == category.casefold()
        ]

    columns = [
        "patient_id",
        "age",
        "age_group",
        "reported_symptoms",
        "synthetic_condition_label",
        "category",
        "synthetic_treatment_category"
    ]

    return df[columns]




# result of the BFS search algorithm on the healthcare graph
if __name__ == "__main__":
    result = bfs(
        HEALTHCARE_GRAPH,
        "Assessment",
        "Outcome"
    )

    print("Algorithm: BFS")
    print("Goal found:", result["goal_found"])
    print("Exploration order:", result["exploration_order"])
    print("Path:", " -> ".join(result["path"]))


    print("Healthcare graph test:")
    test_graph = {
        "Start": ["A"],
        "A": [],
        "Goal": []
    }

    test_result = bfs(test_graph, "Start", "Goal")

    print("\nUnreachable goal test:")
    print("Goal found:", test_result["goal_found"])
    print("Path:", test_result["path"])
    print("Exploration order:", test_result["exploration_order"])


    #DFS search algorithm test
    
    print("\nDFS healthcare graph test:")

    dfs_result = dfs(
        HEALTHCARE_GRAPH,
        "Assessment",
        "Outcome"
    )

    print("Goal found:", dfs_result["goal_found"])
    print("Exploration order:", dfs_result["exploration_order"])
    print("Path:", " -> ".join(dfs_result["path"]))

    #A* search algorithm test
    
    print("\n--- A* Search ---")
    result = a_star(
        HEALTHCARE_GRAPH,
        EDGE_COSTS,
        HEURISTIC,
        "Assessment",
        "Outcome"
    )

    print("Path:", " -> ".join(result["path"]))
    print("Exploration order:", result["exploration_order"])
    print("Goal found:", result["goal_found"])
    print("Total path cost:", result["path_cost"])




    
    print("\n--- Interactive Healthcare Dataset Search ---")

    symptom = input("Enter a symptom: ").strip()

    if not symptom:
        print("Please enter a symptom.")
    else:
        results = search_dataset_by_symptom(symptom)

        print("\nMatching records:", len(results))

        if results.empty:
            print("No matching records found.")
        else:
            print("\nMatching patient records:")
            print(results.to_string(index=False))


            
            print("\nObserved categories:")
            print(results["category"].value_counts().to_string())









