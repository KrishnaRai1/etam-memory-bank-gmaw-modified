import json
import yaml
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict


def parse_eda_annotations(excel_path: Path, output_dir: Path):
    """
    Parses the EDA Excel annotations, extracts frame intervals,
    and generates benchmark configuration files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = defaultdict(list)

    print(f"[Parser] Reading annotations from {excel_path}...")

    # Read Excel file
    df = pd.read_excel(excel_path)

    print(f"[Parser] Loaded {len(df)} rows")

    for idx, row in df.iterrows():

        # Replace NaN with empty strings
        row = row.fillna("")

        try:
            # Adjust these indices if your sheet layout changes
            video_id = str(row.iloc[1]).strip()
            frame_range = str(row.iloc[2]).strip()
            error_type = str(row.iloc[3]).strip()
            notes = str(row.iloc[4]).strip()

            # Optional counts
            manual_count = -1
            system_count = -1

            if len(row) > 8:
                try:
                    manual_count = int(float(row.iloc[8]))
                except:
                    pass

            if len(row) > 9:
                try:
                    system_count = int(float(row.iloc[9]))
                except:
                    pass

            # Skip invalid rows
            if not video_id or not frame_range:
                continue

            # Parse frame range like "8245-8330"
            match = re.search(r"(\d+)\s*-\s*(\d+)", frame_range)

            if not match:
                print(f"[Warning] Invalid frame range at row {idx}: {frame_range}")
                continue

            start_frame = int(match.group(1))
            end_frame = int(match.group(2))

            # Categorize errors
            category = "uncategorized"

            error_lower = error_type.lower()

            if "segmentation" in error_lower:
                category = "segmentation_errors"

            elif "id change" in error_lower or "switch" in error_lower:
                category = "id_switches"

            elif "false droplet" in error_lower or "spatter" in error_lower:
                category = "false_positive_spatter"

            elif "incorrect id" in error_lower:
                category = "incorrect_id_assignment"

            elif "count" in error_lower:
                category = "counting_failures"

            elif "difficult" in error_lower:
                category = "difficult_motion"

            interval_data = {
                "interval_id": f"{video_id}_{start_frame}_{end_frame}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "category": category,
                "error_description": error_type,
                "notes": notes,
                "manual_count": manual_count,
                "system_count": system_count
            }

            benchmark_dict[video_id].append(interval_data)

        except Exception as e:
            print(f"[Error] Failed processing row {idx}: {e}")

    # -----------------------------
    # Save JSON metadata
    # -----------------------------
    json_path = output_dir / "parsed_benchmark_intervals.json"

    with open(json_path, "w") as f:
        json.dump(benchmark_dict, f, indent=2)

    # -----------------------------
    # Save benchmark YAML
    # -----------------------------
    yaml_config = {
        "benchmark_cases": dict(benchmark_dict)
    }

    yaml_path = output_dir / "benchmark_cases.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(
            yaml_config,
            f,
            sort_keys=False,
            default_flow_style=False
        )

    total_intervals = sum(len(v) for v in benchmark_dict.values())

    print("\n========== PARSER SUMMARY ==========")
    print(f"Videos found           : {len(benchmark_dict)}")
    print(f"Total intervals parsed : {total_intervals}")
    print(f"YAML saved             : {yaml_path}")
    print(f"JSON saved             : {json_path}")
    print("====================================")


if __name__ == "__main__":

    excel_input = Path(
        "Exploratory Data Analysis (EDA)_Data Annotation.xlsx"
    )

    out_dir = Path("configs")

    parse_eda_annotations(excel_input, out_dir)