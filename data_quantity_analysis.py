# all imports necessary
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from collections import Counter
from typing import Union

__all__ = [
    "rating_vs_path_length",
    "path_duration_distribution",
    "path_length_vs_duration",
    "path_length_distribution",
    "most_visited_articles",
    "top_100_visited_articles",
    "top_100_target_articles",
    "distribution_position_percentage",
    "count_in_out_neighbors",
    "get_categories_main_article",
    "separate_categories",
    "analyze_articles_near",
    "analyze_nearby_articles_at_different_distances",
    "get_categories_art",
    "combine_results",
]


# Rating vs path length
def rating_vs_path_length(paths_finished: pd.DataFrame, show: bool = False):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 9))

    ax = sns.catplot(
        y="path_length",
        data=paths_finished,
        kind="bar",
        palette="coolwarm",
        hue="rating",
    )
    ax.set_axis_labels("Rating (where 0 = no rating)", "Length")
    ax.despine(left=True)
    plt.title("Length of the path in function of the rating")
    # Save the plot to a file
    plt.savefig("Rating_vs_Path_length.png", bbox_inches="tight")
    if show:
        plt.show()


#################################################################################################


# Path duration distribution - maybe not important
def path_duration_distribution(paths_finished: pd.DataFrame, show: bool = False):
    # Create a histogram for path duration distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(
        paths_finished["durationInSec"],
        bins=100,
        edgecolor="k",
        alpha=0.7,
        log_scale=True,
    )
    plt.yscale("log")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.title("Path Duration Distribution for Finished Paths")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if show:
        plt.show()


#################################################################################################


# Path length versus duration
def path_length_vs_duration(paths_finished: pd.DataFrame, show: bool = False):
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        paths_finished["path_length"],
        paths_finished["durationInSec"],
        alpha=0.7,
    )
    plt.xlabel("Path Length")
    plt.ylabel("Duration (seconds)")
    plt.yscale("log")
    plt.title("Relationship between Path Length and Duration for Finished Paths")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if show:
        plt.show()


#################################################################################################


# Path length distribution
def path_length_distribution(paths_finished: pd.DataFrame, show: bool = False):
    # Create a histogram for path length distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(
        paths_finished["path_length"],
        bins=50,
        edgecolor="k",
        alpha=0.7,
        log_scale=True,
    )
    plt.xlabel("Path Length (Number of Articles)")
    plt.ylabel("Count")
    plt.title("Path Length Distribution for Finished Paths")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if show:
        plt.show()


#################################################################################################


# Most commonly visited articles - not nice plot
def most_visited_articles(paths_finished: pd.DataFrame, show: bool = False):
    # Flatten the list of lists into a single list of visited articles
    flat_visited_articles = [
        article for path in paths_finished["path"] for article in path
    ]

    # Count occurrences using Counter
    article_counts = Counter(flat_visited_articles)

    # Convert the Counter to a dictionary
    article_counts_dict = dict(article_counts)

    # Extract article names and counts
    article_names = list(article_counts_dict.keys())
    article_counts = list(article_counts_dict.values())

    # Create a bar chart with article names on the x-axis
    plt.figure(figsize=(10, 6))
    plt.bar(article_names, article_counts, edgecolor="k", alpha=0.7)
    plt.xlabel("Visited Article")
    plt.ylabel("Count")
    plt.title("Most Commonly Visited Articles")

    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if show:
        plt.show()


#################################################################################################


# Top 100 most visited articles
def top_100_visited_articles(paths_finished: pd.DataFrame, show: bool = False):
    # Flatten the list of lists into a single list of visited articles
    flat_visited_articles = [
        article for path in paths_finished["path"] for article in path
    ]

    # Count occurrences using Counter
    article_counts = Counter(flat_visited_articles)

    # Get the 100 most common articles
    top_100_articles = article_counts.most_common(100)

    print(top_100_articles)
    # Remove the tuple with the first element equal to '<'
    filtered_articles = [
        (name, count) for (name, count) in top_100_articles if name != "<"
    ]
    
    # Extract article names and counts
    article_names, article_counts = zip(*filtered_articles)

    # Create a bar chart with the 100 most visited articles
    plt.figure(figsize=(13, 6))
    sns.barplot(x=article_names, y=article_counts, palette="coolwarm")
    plt.yscale("log")
    plt.xlabel("Visited Article")
    plt.ylabel("Count")
    plt.title("Top 100 Most Visited Articles")
    plt.rc("xtick", labelsize=7)
    plt.rc("ytick", labelsize=7)
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if show:
        plt.show()


#################################################################################################


# Top 100 most common target in finished articles
def top_100_target_articles(paths_finished: pd.DataFrame, show: bool = False):
    # Extract the last article as the target
    paths_finished["target_article"] = paths_finished["path"].str[-1]

    # Count the occurrences of each target article
    target_counts = paths_finished["target_article"].value_counts()

    # Get the top 100 most common target articles
    top_100_targets = target_counts.head(100)

    # Create a bar chart for the top 100 target articles
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_100_targets.index, y=top_100_targets.values, palette="coolwarm")
    plt.yscale("log")
    plt.xlabel("Target Article")
    plt.ylabel("Count")
    plt.title("Top 100 Most Common Target Articles in Finished Paths")
    plt.rc("xtick", labelsize=7)
    plt.rc("ytick", labelsize=7)
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if show:
        plt.show()


#################################################################################################


# Distribution of the position (in %) of the main articles along finished paths that contain it
def distribution_position_percentage(
    paths_finished: pd.DataFrame, main_article: str, show: bool = False
):
    # Define a function to calculate the position percentage of the main article in each path
    def compute_position_percentage(path):
        try:
            return path.index(main_article) / len(path)
        except ValueError:
            return None

    # Apply the function to each path and create a new column for percentage position
    position_percentage = paths_finished["path"].apply(compute_position_percentage)

    # Create a histogram to visualize the distribution of the position percentage of the main article in paths
    plt.figure(figsize=(10, 6))
    sns.histplot(
        position_percentage,
        bins=25,
        edgecolor="k",
        alpha=0.7,
        log_scale=True,
    )
    plt.xlabel("Percentage Position of United Kingdom in Path")
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of the position percentage of {main_article.replace('_',' ')} in Finished Paths"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("Percentage_Pos_UK_Path.png", bbox_inches="tight")
    if show:
        plt.show()


#################################################################################################


def count_in_out_neighbors(
    paths_finished: pd.DataFrame, paths_unfinished: pd.DataFrame, main_article: str
):
    def contains_main_article(path: list[str]):
        return any(article.lower() == main_article.lower() for article in path)

    # 1 number of finished paths that contain the main article
    paths_with_main_article = paths_finished["path"].apply(contains_main_article)
    num_paths_with_main_articles = paths_with_main_article.sum()
    print(
        f"Number of paths finished that contain {main_article.replace('_',' ')}:",
        num_paths_with_main_articles,
    )

    # 2 Number of unfinished paths that contain the main article
    paths_with_main_article = paths_unfinished["path"].apply(contains_main_article)
    num_paths_with_main_articles = paths_with_main_article.sum()
    print(
        f"Number of paths unfinished that contain {main_article.replace('_', '')}",
        num_paths_with_main_articles,
    )

    # 3 Number of finished paths that lead to the main article
    finished_with_main_article = (
        paths_finished["target_article"].str.lower() == main_article.lower()
    )
    num_paths_finished_with_main_article = finished_with_main_article.sum()
    print(
        f"Number of paths that finished with {main_article.replace('_', ' ')}:",
        num_paths_finished_with_main_article,
    )


#################################################################################################


# 1. Categories of UK
def get_categories_main_article(main_article: str, name_cat: str, categories: pd.DataFrame) -> list[str]:
    categories_main_article = categories.loc[categories["article"] == main_article, name_cat]
    categories_main_article = categories_main_article.values.flatten()
    categories_main_article = [cat for cat in categories_main_article if pd.notna(cat)]
    categories_main_article = list(set(categories_main_article))
    return categories_main_article


def separate_categories(
    main_article: str,
    categories: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str]]:
    cat_all = get_categories_main_article(main_article, ["category1", "category2", "category3"], categories)
    # print("All categories:", cat_all)
    subcat1 = get_categories_main_article(main_article, ["category1"], categories)
    # print("Category 1:", cat_1)
    subcat2 = get_categories_main_article(main_article, ["category2"], categories)
    # print("Category 2", subcat2)
    subcat3 = get_categories_main_article(main_article, ["category3"], categories)
    # print("Category 3", subcat3)
    return cat_all, subcat1, subcat2, subcat3


#################################################################################################


# 2. Function that looks at the categories from the articles at one, two, three steps from the 
# main_article, in this milestone it is UK, in the game path
def analyze_articles_near(
    paths_finished: pd.DataFrame,
    main_article: str,
    steps_away: int,
    namecat: Union[str, list[str]],
    categories: pd.DataFrame,
    categories_main_article: list,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    results = []
    n_results = []

    for _, row in paths_finished.iterrows():
        path = row["path"]
        if main_article in path:
            reference_index = path.index(main_article)
            start_index = max(0, reference_index - steps_away)
            end_index = min(len(path), reference_index + steps_away + 1)
            articles_to_analyze = path[start_index:end_index]

            # Exclude the reference article
            articles_to_analyze = [
                article
                for article in articles_to_analyze
                if article != main_article
            ]

            # Collect the categories of the articles to analyze
            categories_of_articles_to_analyze = []
            for article in articles_to_analyze:
                article_categories = categories.loc[
                    categories["article"] == article, namecat
                ]
                article_categories = article_categories.values.flatten()
                article_categories = [
                    cat for cat in article_categories if pd.notna(cat)
                ]
                categories_of_articles_to_analyze.extend(article_categories)

            # Compare the categories with those of UK
            for category in categories_of_articles_to_analyze:
                if category in categories_main_article:
                    results.append((main_article, category))
                else:
                    n_results.append((main_article, category))

    return results, n_results


#################################################################################################


# 3. Analyze articles at one, two, and three steps away from main article
def analyze_nearby_articles_at_different_distances(
    paths_finished: pd.DataFrame,
    main_article: str,
    cat_name: Union[str, list[str]],
    categories: pd.DataFrame,
    cat_all: list[str],
):
    if isinstance(cat_name, str):
        cat_name = [cat_name]
    # Last argument to change with desired category/ies
    results_1_step, non1 = analyze_articles_near(
        paths_finished,
        main_article,
        1,
        cat_name,
        categories,
        cat_all,
    )
    results_2_steps, non2 = analyze_articles_near(
        paths_finished,
        main_article,
        2,
        cat_name,
        categories,
        cat_all,
    )
    results_3_steps, non3 = analyze_articles_near(
        paths_finished,
        main_article,
        3,
        cat_name,
        categories,
        cat_all,
    )

    # Display the results
    # print("Results at 1 step away from UK:", results_1_step)
    print("Non-coincide Categories at 1 step away from UK:", non1[1:10])
    # print("\nResults at 2 steps away from UK:", results_2_steps)
    print("Non-coincide Categories at 2 steps away from UK:", non2[1:10])
    # print("\nResults at 3 steps away from UK:", results_3_steps)
    print("Non-coincide Categories at 3 steps away from UK:", non3[1:10])
    return results_1_step, results_2_steps, results_3_steps, non1, non2, non3


#################################################################################################


# List of categories for a cliché article - here William Shakespeare is presented as an example
def get_categories_art(
    categories: pd.DataFrame,
    cliche_article: str,
    name_cat: Union[str, list[str]] = ["category1", "category2", "category3"],
) -> list[str]:
    categories_art = categories.loc[categories["article"] == cliche_article, name_cat]
    categories_art = categories_art.values.flatten()
    categories_art = [cat for cat in categories_art if pd.notna(cat)]
    categories_art = list(set(categories_art))
    print(categories_art)
    return categories_art


#################################################################################################


# Combine all the results - show graphs for step 1, 2, 3 and combined - away from the article
# in question (Here all articles that present United_Kingdom in their name) and show in a bar plot
# the categories that coicide with thos of the main article "UK" and the ones that do not. 
# Moreover: an arrow indicates the categories of the cliché chosen here above
def combine_results(
    paths_finished: pd.DataFrame,
    main_article: str,
    categories: pd.DataFrame,
    article_categories: list[str],
    namecat: Union[str, list[str]] = ["category1", "category2", "category3"],
):
    if isinstance(namecat, str):
        namecat = [namecat]

    # Last argument to change with desired category/ies
    (
        results_1_step,
        results_2_steps,
        results_3_steps,
        non1,
        non2,
        non3,
    ) = analyze_nearby_articles_at_different_distances(
        paths_finished, main_article, namecat, categories, article_categories
    )
    categories_art = get_categories_art(categories, "William_Shakespeare", namecat)

    # Combine results from all steps
    # William Shakespeare_
    all_results = results_1_step + results_2_steps + results_3_steps
    all_non = non1 + non2 + non3

    # Count the occurrences of each category for the height of the bars in the barplot
    def cat_count(res):
        category_counts = {}
        for _, category in res:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        return category_counts

    # Separate the categories and their counts
    category_counts_all = cat_count(all_results)

    # non-coincide categories
    category_ncounts_all = cat_count(all_non)

    # Combine coincide and non-coincide counts for each step
    combined_counts_all = {
        cat: category_counts_all.get(cat, 0) for cat in set(category_counts_all)
    }
    combined_ncounts_all = {
        cat: category_ncounts_all.get(cat, 0) for cat in set(category_ncounts_all)
    }

    # Create a combined bar plot for all steps
    categories_list_all = sorted(
        list(set(combined_counts_all.keys()) | set(combined_ncounts_all.keys()))
    )
    counts_all = [combined_counts_all.get(cat, 0) for cat in categories_list_all]
    ncounts_all = [combined_ncounts_all.get(cat, 0) for cat in categories_list_all]
    index_all = np.arange(len(categories_list_all))

    # Create a list of colors where highlighted categories are in a different color
    # this helps position the arrow n the right bar for the plots below
    # index_all = np.arange(len(categories_list_all))
    bar_width = 0.35
    bar_positions = index_all + bar_width / 2

    plt.figure(figsize=(12, 6))
    plt.bar(index_all, counts_all, color="skyblue", label="Coincide with UK")
    plt.bar(
        index_all + bar_width,
        ncounts_all,
        color="lightcoral",
        label="Do not coincide with UK",
    )
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Category Counts for All Steps")
    plt.xticks(index_all + bar_width / 2, categories_list_all, rotation=90, ha="right")
    # plt.ylim(0, max(counts_all) + 500)
    plt.legend()
    plt.rc("xtick", labelsize=6)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Add arrows above bars corresponding to article_categories
    for category in categories_art:
        if category in categories_list_all:
            category_index = categories_list_all.index(category)
            arrow_position = bar_positions[category_index] + 0.5 * bar_width

            # Check if the arrow will overlap with neighboring bars
            if (
                category_index > -1
                and arrow_position - -1.1 < bar_positions[category_index - 1]
            ):
                arrow_position = bar_positions[category_index - 0] + 0.1
            elif (
                category_index < len(categories_list_all) - 0
                and arrow_position + -1.1 > bar_positions[category_index]
            ):
                arrow_position = bar_positions[category_index + 0] - 0.1

            plt.annotate(
                "v",
                xy=(
                    arrow_position,
                    max(counts_all[category_index], ncounts_all[category_index]) + 49,
                ),
                ha="center",
                va="bottom",
                color="red",
                fontsize=11,
            )

    # Save the plot to a file
    # plt.savefig("all_steps_plot_combined.png", bbox_inches="tight")
    plt.show()


#################################################################################################
