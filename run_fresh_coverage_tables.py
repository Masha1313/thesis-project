import argparse

from molecular_similarity_experiment import (
    build_all_coverage_summary,
    build_correlation_summary,
    coverage_pretty,
    run_multiple_seeds,
    summarize_coverage,
)


def write_coverage_outputs(df_all_cov, output_prefix: str) -> None:
    df_results, mean_all, mean_at_least_one = summarize_coverage(df_all_cov)

    df_results.to_csv(f"{output_prefix}_coverage_results.csv", index=False)

    outputs = [
        ("all", mean_all),
        ("at_least_one", mean_at_least_one),
    ]

    for label, mean_df in outputs:
        mean_df.to_csv(f"{output_prefix}_coverage_mean_{label}.csv", index=False)
        pretty = coverage_pretty(mean_df)
        pretty.to_csv(f"{output_prefix}_coverage_mean_{label}_pretty.csv", index=False)
        pretty.to_csv(f"{output_prefix}_table_{label}.csv", index=False)

        print(f"\n{label}:")
        print(pretty.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="chembl_raw_dump.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--target-n", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--grid-width", type=float, default=0.2)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--output-prefix", default="fresh")
    args = parser.parse_args()

    all_seed_results = run_multiple_seeds(
        csv_file=args.csv,
        seeds=args.seeds,
        target_n=args.target_n,
        grid_width=args.grid_width,
        n_runs=args.n_runs,
        max_order=args.max_order,
    )

    df_all_cov = build_all_coverage_summary(all_seed_results)
    write_coverage_outputs(df_all_cov, args.output_prefix)

    corr = build_correlation_summary(all_seed_results)
    corr.to_csv(f"{args.output_prefix}_correlation_summary.csv", index=False)


if __name__ == "__main__":
    main()
