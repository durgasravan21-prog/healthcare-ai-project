
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

from data_quality import load_data as load_quality_data
from data_quality import check_data_quality, print_quality_report


from analysis import (
    load_healthcare_data as load_analysis_data,
    plot_age_category_heatmap,
)
from report import load_dataset, generate_report

def show_summary(df):
    knowledge = build_knowledge_base(df)

    print("\n=== Healthcare Dataset Summary ===")
    print("Total records:", knowledge["total_records"])
    print("Average age:", knowledge["average_age"])

    print("\nRecords by age group:")
    for group, count in knowledge["age_group_counts"].items():
        print(f"{group}: {count}")

    print("\nRecords by category:")
    for category, count in knowledge["category_counts"].items():
        print(f"{category}: {count}")


def show_reasoning(df):
    knowledge = build_knowledge_base(df)
    observations = generate_observations(knowledge)

    print("\n=== Rule-Based Observations ===")
    for number, observation in enumerate(observations, start=1):
        print(f"{number}. {observation}")

    print("\nThese observations describe synthetic data only.")



def search_records():
    from search import filter_healthcare_records

    print("\n=== Advanced Healthcare Record Search ===")
    symptom = input("Symptom (press Enter to skip): ").strip()
    age_group = input(
        "Age group - Child, Adult, Senior (Enter to skip): "
    ).strip()
    category = input(
        "Category (Enter to skip): "
    ).strip()

    results = filter_healthcare_records(
        symptom=symptom,
        age_group=age_group,
        category=category
    )

    print("\nMatching records:", len(results))

    if results.empty:
        print("No matching records found.")
    else:
        print(results.to_string(index=False))



def show_charts(df):
    print("\nOpening age-group chart...")
    plot_age_group_distribution(df)

    print("Opening category chart...")
    plot_category_distribution(df)

    print("Opening treatment-category chart...")
    plot_treatment_distribution(df)
def show_data_quality():
    data = load_quality_data()
    report = check_data_quality(data)
    print_quality_report(report)

def show_heatmap():
    df = load_analysis_data()
    print("Generating age-group and category heatmap...")
    plot_age_category_heatmap(df)


def show_project_report():
    df = load_dataset()
    generate_report(df)


def main():
    df = load_healthcare_data()

    while True:
        print("\n=== Healthcare Records Analysis ===")
        print("1. View dataset summary")
        print("2. View rule-based observations")
        print("3. Search records by symptom")
        print("4. Display charts")
        print("5. Run data-quality checks")
        print("6. Display age-category heatmap")
        print("7. Generate project report")
        print("8. Exit")

        choice = input("Choose an option (1-8): ").strip()

        if choice == "1":
            show_summary(df)
        elif choice == "2":
            show_reasoning(df)
        elif choice == "3":
            search_records()
        elif choice == "4":
            show_charts(df)
        elif choice == "5":
            show_data_quality()
        elif choice == "6":
            show_heatmap()
        elif choice == "7":
            show_project_report()
        elif choice == "8":
            print("Exiting the application.")
            break
        else:
            print("Invalid option. Please enter a number from 1 to 8.")


if __name__ == "__main__":
    main()
