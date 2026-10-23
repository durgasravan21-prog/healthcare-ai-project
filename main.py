
from pathlib import Path

import pandas as pd

from src.knowledge_base import (
    load_healthcare_data,
    build_knowledge_base,
)
from src.reasoning import generate_observations
from src.visualization import (
    plot_age_group_distribution,
    plot_category_distribution,
    plot_treatment_distribution,
)

from data_quality import (
    load_data as load_quality_data,
    check_data_quality,
    print_quality_report,
)

from analysis import plot_age_category_heatmap
from report import load_dataset, generate_report

from preprocessing import main as run_preprocessing
from model import main as run_model
from evaluation import evaluate_model
from responsible_ai import main as run_responsible_ai


PROJECT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# 1. DATASET SUMMARY
# --------------------------------------------------

def show_summary(df):
    knowledge = build_knowledge_base(df)

    print("\n=== HEALTHCARE DATASET SUMMARY ===")
    print("Total records:", knowledge["total_records"])
    print("Average age:", knowledge["average_age"])

    print("\nRecords by age group:")
    for group, count in knowledge["age_group_counts"].items():
        print(f"{group}: {count}")

    print("\nRecords by category:")
    for category, count in knowledge["category_counts"].items():
        print(f"{category}: {count}")


# --------------------------------------------------
# 2. KNOWLEDGE-BASE REASONING
# --------------------------------------------------

def show_reasoning(df):
    knowledge = build_knowledge_base(df)
    observations = generate_observations(knowledge)

    print("\n=== RULE-BASED OBSERVATIONS ===")

    if isinstance(observations, (list, tuple)):
        for number, observation in enumerate(observations, start=1):
            print(f"{number}. {observation}")
    else:
        print(observations)

    print("\nThese observations describe synthetic data only.")


# --------------------------------------------------
# 3. SEARCH AND FILTER RECORDS
# --------------------------------------------------

def search_records():
    from search import filter_healthcare_records

    print("\n=== ADVANCED HEALTHCARE RECORD SEARCH ===")
    print("Leave a field blank to skip that filter.")

    symptom = input("Symptom: ").strip()
    age_group = input("Age group (Child/Adult/Senior): ").strip()
    category = input("Category: ").strip()

    results = filter_healthcare_records(
        symptom=symptom,
        age_group=age_group,
        category=category,
    )

    print("\nMatching records:", len(results))

    if results.empty:
        print("No matching records found.")
    else:
        print(results.head(20).to_string(index=False))

        if len(results) > 20:
            print("Showing the first 20 results.")


# --------------------------------------------------
# 4. VISUALIZATION
# --------------------------------------------------

def show_charts(df):
    knowledge = build_knowledge_base(df)

    print("\nGenerating age-group distribution chart...")
    plot_age_group_distribution(knowledge)

    print("\nGenerating category distribution chart...")
    plot_category_distribution(knowledge)

    print("\nGenerating treatment-category distribution chart...")
    plot_treatment_distribution(knowledge)

    print("\nChart functions completed.")


def show_heatmap(df):
    print("\nGenerating age-group and category heatmap...")
    plot_age_category_heatmap(df)


# --------------------------------------------------
# 5. DATA QUALITY
# --------------------------------------------------

def show_data_quality():
    data = load_quality_data()
    quality_report = check_data_quality(data)
    print_quality_report(quality_report)


# --------------------------------------------------
# 6. PROJECT REPORT
# --------------------------------------------------

def show_project_report():
    df = load_dataset()
    generate_report(df)


# --------------------------------------------------
# 7. BFS AND DFS DEMONSTRATION
# --------------------------------------------------


def demonstrate_search_algorithms():
    import heapq
    from collections import deque
    from itertools import count
    from search import HEALTHCARE_GRAPH

    def bfs(graph, start, goal):
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            node, path = queue.popleft()

            if node == goal:
                return path

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def dfs(graph, start, goal):
        stack = [(start, [start])]
        visited = set()

        while stack:
            node, path = stack.pop()

            if node == goal:
                return path

            if node in visited:
                continue

            visited.add(node)

            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None

    def astar(graph, start, goal):
        # Calculate shortest remaining distances for the heuristic.
        # All graph edges have illustrative cost 1.
        reverse_graph = {node: [] for node in graph}

        for node, neighbors in graph.items():
            for neighbor in neighbors:
                reverse_graph.setdefault(neighbor, []).append(node)

        distance = {goal: 0}
        queue = deque([goal])

        while queue:
            node = queue.popleft()

            for previous in reverse_graph.get(node, []):
                if previous not in distance:
                    distance[previous] = distance[node] + 1
                    queue.append(previous)

        if start not in distance:
            return None, None

        # Priority queue entries: (f score, tie-breaker, g score, node, path)
        counter = count()
        frontier = [
            (distance[start], next(counter), 0, start, [start])
        ]
        best_cost = {start: 0}

        while frontier:
            _, _, cost, node, path = heapq.heappop(frontier)

            if cost != best_cost.get(node):
                continue

            if node == goal:
                return path, cost

            for neighbor in graph.get(node, []):
                new_cost = cost + 1

                if new_cost < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = new_cost
                    heuristic = distance.get(neighbor, float("inf"))

                    heapq.heappush(
                        frontier,
                        (
                            new_cost + heuristic,
                            next(counter),
                            new_cost,
                            neighbor,
                            path + [neighbor],
                        ),
                    )

        return None, None

    print("\n=== BFS, DFS AND A* SEARCH ===")
    print("Available nodes:")
    print(", ".join(HEALTHCARE_GRAPH.keys()))

    start = input(
        "Starting node (Enter for Assessment): "
    ).strip() or "Assessment"

    goal = input(
        "Goal node (Enter for Outcome): "
    ).strip() or "Outcome"

    if start not in HEALTHCARE_GRAPH:
        print("Invalid starting node.")
        return

    if goal not in HEALTHCARE_GRAPH:
        print("Invalid goal node.")
        return

    print("\nBFS path:", bfs(HEALTHCARE_GRAPH, start, goal))
    print("DFS path:", dfs(HEALTHCARE_GRAPH, start, goal))

    path, cost = astar(HEALTHCARE_GRAPH, start, goal)
    print("A* path:", path)
    print("A* path cost:", cost)

    print(
        "\nNote: This is an educational graph-search demonstration. "
        "Edge costs are illustrative and do not represent medical costs "
        "or treatment recommendations."
    )





    def dfs(graph, start, goal):
        stack = [(start, [start])]
        visited = set()

        while stack:
            node, path = stack.pop()

            if node == goal:
                return path

            if node in visited:
                continue

            visited.add(node)

            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None

    print("\n=== SEARCH ALGORITHMS ===")
    print("Available nodes:")
    print(", ".join(HEALTHCARE_GRAPH.keys()))

    start = input(
        "Starting node (press Enter for Assessment): "
    ).strip() or "Assessment"

    goal = input(
        "Goal node (press Enter for Outcome): "
    ).strip() or "Outcome"

    if start not in HEALTHCARE_GRAPH:
        print("Invalid starting node.")
        return

    if goal not in HEALTHCARE_GRAPH:
        print("Invalid goal node.")
        return

    print("\nBFS path:", bfs(HEALTHCARE_GRAPH, start, goal))
    print("DFS path:", dfs(HEALTHCARE_GRAPH, start, goal))

    print(
        "\nThese algorithms demonstrate pathfinding on an "
        "illustrative workflow, not medical recommendations."
    )

def show_feature_importance():
    from feature_importance import main as run_feature_importance

    run_feature_importance()


# --------------------------------------------------
# 8. MAIN MENU
# --------------------------------------------------

def display_menu():
    print("\n" + "=" * 58)
    print(" HEALTHCARE RECORDS VISUALIZATION AND ANALYSIS")
    print("=" * 58)
    print(" 1. Dataset summary")
    print(" 2. Knowledge-base reasoning")
    print(" 3. Search and filter records")
    print(" 4. Display charts")
    print(" 5. Data-quality checks")
    print(" 6. Age-category heatmap")
    print(" 7. Generate project report")
    print(" 8. Preprocess dataset")
    print(" 9. Train machine-learning model")
    print("10. Evaluate machine-learning model")
    print("11. Generate Responsible AI report")
    print("12. Demonstrate BFS and DFS")
    print("13. Display model feature importance")
    print("14. Exit")


def main():
    while True:
        display_menu()
        choice = input("Choose an option (1-13): ").strip()

        try:
            if choice == "1":
                df = load_healthcare_data()
                show_summary(df)

            elif choice == "2":
                df = load_healthcare_data()
                show_reasoning(df)

            elif choice == "3":
                search_records()

            elif choice == "4":
                df = load_healthcare_data()
                show_charts(df)

            elif choice == "5":
                show_data_quality()

            elif choice == "6":
                df = load_healthcare_data()
                show_heatmap(df)

            elif choice == "7":
                show_project_report()

            elif choice == "8":
                run_preprocessing()

            elif choice == "9":
                run_model()

            elif choice == "10":
                evaluate_model()

            elif choice == "11":
                run_responsible_ai()

            elif choice == "12":
                demonstrate_search_algorithms()

            elif choice == "13":
                print("\nExiting project. Goodbye!")
                break
            elif choice == "14":
                show_feature_importance()

            else:
                print("\nInvalid choice. Enter a number from 1 to 14.")
                continue

        except Exception as error:
            print(f"\nAn error occurred: {error}")
            print("Check the relevant module and its file paths.")

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
