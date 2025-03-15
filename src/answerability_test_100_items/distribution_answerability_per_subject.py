import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_json(filename):
    """Load JSON file into a Pandas DataFrame."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def process_data(source1_file, source2_file, output_image):
    # Load the JSON data into Pandas DataFrames
    df1 = load_json(source1_file)
    df2 = load_json(source2_file)

    # Perform an inner join on the "id" key
    merged_df = df1.merge(df2, on="id", how="inner")

    # Aggregate correctness by subject
    summary_df = merged_df.groupby("subject_name")["is_correct"].value_counts(normalize=True).unstack(fill_value=0) * 100

    # Renaming columns for clarity
    summary_df.columns = ["Incorrect (%)", "Correct (%)"]

    # Display summary table
    print(summary_df)

    # Plot the stacked bar chart
    summary_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap="viridis")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Subject Name")
    plt.title("Distribution of Correct and Incorrect Answers by Subject")
    plt.legend(title="Answer Accuracy")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot instead of displaying it
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    print(f"Chart saved as {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join two JSON files and visualize correctness distribution.")
    parser.add_argument("source1", help="Path to the first JSON file (source1.json)")
    parser.add_argument("source2", help="Path to the second JSON file (source2.json)")
    parser.add_argument("output_image", help="Filename for the output image (e.g., output.png)")

    args = parser.parse_args()

    process_data(args.source1, args.source2, args.output_image)
