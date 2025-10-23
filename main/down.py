# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:57:08 2025

@author: csaw7183
"""
# %%
import pandas as pd
import plotly.express as px
import os

# %%

def count_colonies_per_plate(combined_csv_path, min_score=0.0):
    """
    Reads a combined colonies CSV and returns a DataFrame
    with counts of colonies per Plate, considering only rows
    where 'Score' > min_score.

    If min_score is a list, counts are calculated for each score threshold
    and returned in additional columns.

    Parameters:
        combined_csv_path (str): Path to the combined CSV file.
        min_score (float or list of floats): Minimum score threshold(s) to include a row.

    Returns:
        pd.DataFrame: A DataFrame with 'Plate' and 'ColonyCount' columns.
                      If min_score is a list, additional columns are added for each threshold.
    """
    df = pd.read_csv(combined_csv_path)

    if "Plate" not in df.columns:
        raise ValueError("The CSV must contain a 'Plate' column.")
    if "Score" not in df.columns:
        raise ValueError("The CSV must contain a 'Score' column.")

    # Ensure min_score is a list for consistent processing
    if isinstance(min_score, (int, float)):
        min_score = [min_score]

    result_df = pd.DataFrame({"Plate": df["Plate"].unique()})
    result_df.sort_values("Plate", inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    # Count colonies for each score threshold
    for score in min_score:
        filtered_df = df[df["Score"] > score]
        counts = filtered_df["Plate"].value_counts().reset_index()
        counts.columns = ["Plate", f"ColonyCount_{score}"]
        result_df = result_df.merge(counts, on="Plate", how="left")

    # Fill NaN with 0 (plates with no colonies above a threshold)
    result_df.fillna(0, inplace=True)
    # Convert counts to integers
    for score in min_score:
        result_df[f"ColonyCount_{score}"] = result_df[f"ColonyCount_{score}"].astype(int)

    return result_df

# %%

def count_colonies_per_plate_and_stripe(
    combined_csv_path,
    min_score=0.0,
    stripe_col="Stripe",
    save_csv=False
):
    """
    Reads a combined colonies CSV and returns a DataFrame where each row
    represents a Plate and each column corresponds to a stripe (or the column
    specified by `stripe_col`). The values represent the number of colonies
    (rows) with Score > min_score for that Plate and stripe.
    The final column is the sum of colonies across all stripes.

    If min_score is a list, separate tables are generated for each threshold
    and concatenated side by side.

    Parameters:
        combined_csv_path (str): Path to the combined CSV file.
        min_score (float or list of floats): Minimum score threshold(s).
        stripe_col (str): Name of the column representing the stripe/grouping
                          variable. Default is "Stripe".
        save_csv (bool): If True, saves the resulting DataFrame as a CSV in
                         the same folder as the input file.

    Returns:
        pd.DataFrame: A DataFrame with Plate as rows, stripe_col as columns,
                      and a final 'Total' column for each threshold.
    """
    df = pd.read_csv(combined_csv_path)

    if "Plate" not in df.columns:
        raise ValueError("The CSV must contain a 'Plate' column.")
    if stripe_col not in df.columns:
        raise ValueError(f"The CSV must contain a '{stripe_col}' column.")
    if "Score" not in df.columns:
        raise ValueError("The CSV must contain a 'Score' column.")

    # Ensure min_score is iterable
    if isinstance(min_score, (int, float)):
        min_score = [min_score]

    result_dfs = []

    for score in min_score:
        # Filter by score
        filtered = df[df["Score"] > score]

        # Count colonies per Plate × stripe_col
        counts = (
            filtered.groupby(["Plate", stripe_col])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Add a total column
        total_col_name = f"Total_{score}"
        counts[total_col_name] = counts.drop(columns="Plate").sum(axis=1)

        # Rename stripe columns to indicate threshold if multiple thresholds are used
        if len(min_score) > 1:
            counts = counts.rename(
                columns={col: f"{col}_>{score}" for col in counts.columns if col not in ["Plate"]}
            )

        result_dfs.append(counts)

    # Merge results for multiple thresholds if needed
    result_df = result_dfs[0]
    for extra_df in result_dfs[1:]:
        result_df = result_df.merge(extra_df, on="Plate", how="outer")

    # Fill NaNs with 0 and ensure integer counts
    result_df = result_df.fillna(0).sort_values("Plate").reset_index(drop=True)
    for col in result_df.columns:
        if col != "Plate":
            result_df[col] = result_df[col].astype(int)

    # Optionally save to CSV
    if save_csv:
        input_dir = os.path.dirname(combined_csv_path)
        input_name = os.path.splitext(os.path.basename(combined_csv_path))[0]
        output_path = os.path.join(input_dir, f"{input_name}_counts.csv")
        result_df.to_csv(output_path, index=False)
        print(f"✅ Output saved to: {output_path}")

    return result_df

# %%

def validate(combined_csv_path = './demo/results/all_results.csv', ref = "ref.csv", min_score=0.97):
    x = count_colonies_per_plate(combined_csv_path, min_score)
    x['Plate'] = x['Plate'].str.replace(r'^(.*)_t\d*(\d)$', r'\1_\2', regex=True)
    print(x)
    
    ref_df = pd.read_csv(ref, delimiter=';')
    merged_df = pd.merge(x, ref_df, on="Plate", how="inner")
    
    subset = merged_df[["Plate", "ColonyCount", "Manual V", "Manual A"]]
    melted = subset.melt(id_vars="Plate", var_name="Metric", value_name="Value")

    # Create interactive barplot
    fig = px.bar(
       melted,
       x="Plate",
       y="Value",
       color="Metric",
       barmode="group",
       title="Colony Metrics per Plate",
       labels={"Value": "Count", "Plate": "Plate", "Metric": "Metric"},
       template="plotly_white"
       )

    output_path = os.path.join("./demo/results", "validation.html")
    fig.write_html(output_path)

    print(f"✅ Plot saved to: {output_path}")

