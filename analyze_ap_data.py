from pathlib import Path

import pandas as pd


def analyze_ap_data(csv_path: Path):
    """
    Analyze AP data to determine appropriate filtering thresholds
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Calculate statistics for each AP at each location
    ap_stats = df.groupby(['location', 'bssid'])['dbm'].agg([
        'count',
        'mean',
        'std',
        'min',
        'max'
    ]).reset_index()

    # Print overall statistics
    print("\nOverall Statistics:")
    print("-" * 50)
    print(f"Total measurements: {len(df)}")
    print(f"Unique locations: {df['location'].nunique()}")
    print(f"Unique BSSIDs: {df['bssid'].nunique()}")

    # Print statistics about counts
    print("\nSample Count Statistics:")
    print("-" * 50)
    count_stats = ap_stats['count'].describe()
    print(count_stats)

    # Print statistics about standard deviations
    print("\nStandard Deviation Statistics:")
    print("-" * 50)
    std_stats = ap_stats['std'].describe()
    print(std_stats)

    # Show distribution of counts
    print("\nCount Distribution:")
    print("-" * 50)
    count_bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, float('inf')]
    count_dist = pd.cut(ap_stats['count'], bins=count_bins).value_counts().sort_index()
    print(count_dist)

    # Show distribution of standard deviations
    print("\nStandard Deviation Distribution:")
    print("-" * 50)
    std_bins = [0, 1, 2, 3, 4, 5, 10, 15, 20, float('inf')]
    std_dist = pd.cut(ap_stats['std'].dropna(), bins=std_bins).value_counts().sort_index()
    print(std_dist)

    # Calculate impact of different thresholds
    print("\nImpact of Different Thresholds:")
    print("-" * 50)
    thresholds = [
        {'min_samples': 2, 'max_std': 20},
        {'min_samples': 3, 'max_std': 15},
        {'min_samples': 4, 'max_std': 10},
        {'min_samples': 5, 'max_std': 8},
    ]

    for thresh in thresholds:
        filtered_aps = ap_stats[
            (ap_stats['count'] >= thresh['min_samples']) &
            (ap_stats['std'].notna()) &
            (ap_stats['std'] <= thresh['max_std'])
            ]
        coverage = len(filtered_aps) / len(ap_stats) * 100
        print(f"Min samples: {thresh['min_samples']}, Max STD: {thresh['max_std']}:")
        print(f"  Retained APs: {len(filtered_aps)} ({coverage:.1f}% of total)")
        print(f"  Unique locations covered: {filtered_aps['location'].nunique()}")
        print(f"  Unique BSSIDs: {filtered_aps['bssid'].nunique()}")
        print()

    return ap_stats


def main():
    # Load and analyze the data
    csv_path = Path.cwd() / 'data' / 'processed' / 'merged_dataset.csv'
    ap_stats = analyze_ap_data(csv_path)

    # Save detailed statistics
    output_dir = Path.cwd() / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    ap_stats.to_csv(output_dir / 'ap_statistics.csv', index=False)


if __name__ == "__main__":
    main()
