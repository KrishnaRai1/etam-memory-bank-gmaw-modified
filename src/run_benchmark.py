import yaml
import json
import csv
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from src.pipeline import run_pipeline

def export_runtime_summary(log_data_list: list[dict], out_dir: Path):
    """Generates an aggregated Markdown report and CSV for benchmark evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Export CSV
    csv_path = out_dir / "benchmark_evaluation_summary.csv"
    headers = [
        "interval_id", "category", "frame_count", "system_droplet_count", 
        "manual_count", "count_delta", "total_runtime_sec", "stage3_runtime_sec"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for log in log_data_list:
            man_count = log.get("benchmark_manual_count", -1)
            sys_count = log.get("droplet_count", 0)
            delta = abs(man_count - sys_count) if man_count != -1 else "N/A"
            
            writer.writerow({
                "interval_id": log.get("benchmark_interval", "unknown"),
                "category": log.get("benchmark_category", "unknown"),
                "frame_count": log.get("frame_count", 0),
                "system_droplet_count": sys_count,
                "manual_count": man_count if man_count != -1 else "N/A",
                "count_delta": delta,
                "total_runtime_sec": round(log.get("total_runtime", 0), 2),
                "stage3_runtime_sec": round(log.get("stage3_runtime", 0), 2)
            })

    # Export Markdown
    md_path = out_dir / "benchmark_evaluation_summary.md"
    categories = Counter([log.get("benchmark_category", "unknown") for log in log_data_list])
    
    with open(md_path, 'w') as f:
        f.write("# Welding Pipeline Benchmark Evaluation\n\n")
        f.write("## Overview\n")
        f.write(f"- **Total Difficult Intervals Processed:** {len(log_data_list)}\n")
        for cat, count in categories.items():
            f.write(f"  - `{cat}`: {count}\n")
            
        f.write("\n## Detailed Results\n")
        f.write("| Interval ID | Category | Frames | Sys Count | Manual Count | Stage 3 Time (s) |\n")
        f.write("|-------------|----------|--------|-----------|--------------|------------------|\n")
        for log in log_data_list:
            man = log.get("benchmark_manual_count", -1)
            man_str = str(man) if man != -1 else "N/A"
            f.write(f"| {log.get('benchmark_interval')} | {log.get('benchmark_category')} | {log.get('frame_count')} | "
                    f"{log.get('droplet_count')} | {man_str} | {round(log.get('stage3_runtime', 0), 2)} |\n")
                    
    print(f"[Benchmark] Generated summary reports at:\n  - {csv_path}\n  - {md_path}")

def execute_benchmarks(pipeline_cfg_path: str, benchmark_yaml_path: str):
    """
    Orchestrates the interval-based execution of the tracking pipeline.
    """
    with open(pipeline_cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)
        
    with open(benchmark_yaml_path, "r") as f:
        benchmarks = yaml.safe_load(f).get("benchmark_cases", {})

    output_root = Path(base_cfg["data"]["output_root"]) / "benchmarks"
    base_video_dir = Path(base_cfg["data"]["video_dir"])
    
    all_metrics = []

    print(f"[Benchmark] Initiating evaluation for {len(benchmarks)} videos...")

    for video_id, intervals in benchmarks.items():
        # Resolve the specific video directory based on the ID
        # Assumes frames are stored like: test_frames/AIS26T1/*.jpg
        video_dir = base_video_dir.parent / video_id
        if not video_dir.exists():
            print(f"[WARN] Video directory {video_dir} not found. Skipping {video_id}.")
            continue
            
        base_cfg["data"]["video_dir"] = str(video_dir)

        print(f"\n[Benchmark] Processing {video_id} ({len(intervals)} intervals)")
        
        for interval in tqdm(intervals, desc=f"Evaluating {video_id}"):
            interval_id = interval["interval_id"]
            start_f = interval["start_frame"]
            end_f = interval["end_frame"]
            
            # Pad the interval slightly for Stage 2 temporal IoU stability
            pad = int(base_cfg.get("stage2", {}).get("window", 5))
            safe_start = max(0, start_f - pad)
            
            # Isolated run directory categorizing the failure mode
            run_dir = output_root / interval["category"] / interval_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Inject metadata so pipeline.py can log it in experiment_logs
            base_cfg["benchmark_meta"] = interval
            
            try:
                # Trigger the pipeline on the specific slice
                out_dir = run_pipeline(
                    cfg=base_cfg,
                    frame_start=safe_start,
                    frame_end=end_f + pad,
                    force_run_dir=run_dir
                )
                
                # Retrieve the generated experiment log to aggregate stats
                log_dir = out_dir / base_cfg.get("stage3", {}).get("experiment_log_dir", "experiment_logs")
                log_files = sorted(log_dir.glob("experiment_*.json"))
                
                if log_files:
                    with open(log_files[-1], 'r') as lf:
                        metrics = json.load(lf)
                        all_metrics.append(metrics)
                else:
                    print(f"[WARN] No experiment log found for {interval_id}.")
                    
            except Exception as e:
                print(f"\n[ERROR] Pipeline execution failed on interval {interval_id}: {str(e)}")

    print("\n[Benchmark] All intervals processed. Aggregating results...")
    export_runtime_summary(all_metrics, output_root)

if __name__ == "__main__":
    PIPELINE_CFG = "configs/pipeline.yaml"
    BENCHMARK_CFG = "configs/benchmark_cases.yaml"
    
    # Ensure the yaml exists first
    if not Path(BENCHMARK_CFG).exists():
        print(f"Error: {BENCHMARK_CFG} not found. Run tools/benchmark_parser.py first.")
    else:
        execute_benchmarks(PIPELINE_CFG, BENCHMARK_CFG)