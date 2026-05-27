import csv
import json
import yaml
import re
from pathlib import Path
from collections import defaultdict

def parse_eda_annotations(csv_path: Path, output_dir: Path):
    """
    Parses the EDA Excel/CSV annotations, extracts frame intervals, 
    and generates the benchmark configuration files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dict = defaultdict(list)
    
    print(f"[Parser] Reading annotations from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers_skipped = False
        
        for row in reader:
            # Skip empty rows or header rows
            if not row or "Frame_number" in row:
                continue
                
            # Based on the provided CSV structure, the columns are offset by an empty first column:
            # [0]: empty
            # [1]: video_id
            # [2]: Frame_number
            # [3]: error type
            # [4]: notes
            # [7]: video id (duplicate)
            # [8]: manually counted droplet
            # [9]: system counted droplet
            
            if len(row) < 4:
                continue
                
            video_id = str(row[1]).strip()
            frame_range = str(row[2]).strip()
            error_type = str(row[3]).strip()
            
            if not video_id or not frame_range:
                continue
                
            # Extract counts if the row has enough columns and they are digits
            manual_count = -1
            sys_count = -1
            if len(row) >= 10:
                if str(row[8]).strip().isdigit():
                    manual_count = int(str(row[8]).strip())
                if str(row[9]).strip().isdigit():
                    sys_count = int(str(row[9]).strip())
                    
            # Parse frame range (e.g., "8245-8330")
            match = re.search(r"(\d+)\s*-\s*(\d+)", frame_range)
            if not match:
                continue
                
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            
            # Categorize the error based on keyword matching
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
                
            interval_data = {
                "interval_id": f"{video_id}_{start_frame}_{end_frame}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "category": category,
                "error_description": error_type,
                "manual_count": manual_count,
                "system_count": sys_count
            }
            
            benchmark_dict[video_id].append(interval_data)

    # 1. Export parsed JSON summary (Rich metadata)
    json_path = output_dir / "parsed_benchmark_intervals.json"
    with open(json_path, 'w') as f:
        json.dump(benchmark_dict, f, indent=2)
        
    # 2. Export benchmark_cases.yaml (Orchestrator configuration)
    yaml_config = {"benchmark_cases": dict(benchmark_dict)}
    yaml_path = output_dir / "benchmark_cases.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, sort_keys=False, default_flow_style=False)
        
    total_intervals = sum(len(v) for v in benchmark_dict.values())
    print(f"[Parser] Success! Extracted {total_intervals} intervals across {len(benchmark_dict)} videos.")
    print(f"[Parser] Saved YAML configuration to: {yaml_path}")
    print(f"[Parser] Saved JSON metadata to: {json_path}")

if __name__ == "__main__":
    # Point this to where your CSV is located
    csv_input = Path("Exploratory Data Analysis (EDA)_Data Annotation.xlsx - Sheet1.csv")
    out_dir = Path("configs")
    parse_eda_annotations(csv_input, out_dir)