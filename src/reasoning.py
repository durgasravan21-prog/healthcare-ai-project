
from .knowledge_base import load_healthcare_data, build_knowledge_base


def generate_observations(knowledge):
    observations = []

    total = knowledge["total_records"]
    age_groups = knowledge["age_group_counts"]
    categories = knowledge["category_counts"]
    treatments = knowledge["treatment_category_counts"]

    if total > 0:
        observations.append(
            f"The dataset contains {total} synthetic records."
        )

    if age_groups:
        largest_age_group = max(
            age_groups, key=age_groups.get
        )
        observations.append(
            f"The largest age group is {largest_age_group} "
            f"with {age_groups[largest_age_group]} records."
        )

    if categories:
        top_category = max(
            categories, key=categories.get
        )
        observations.append(
            f"The most frequent category is {top_category} "
            f"with {categories[top_category]} records."
        )

    if treatments:
        observations.append(
            f"There are {len(treatments)} distinct "
            "synthetic treatment categories."
        )

    return observations


if __name__ == "__main__":
    df = load_healthcare_data()
    knowledge = build_knowledge_base(df)

    print("=== Rule-Based Reasoning ===")

    observations = generate_observations(knowledge)

    for number, observation in enumerate(observations, start=1):
        print(f"{number}. {observation}")

    print(
        "\nNote: These observations describe synthetic data "
        "and are not medical advice."
    )
