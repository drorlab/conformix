"""
generate_report.py

Generate HTML and Markdown reports from benchmark metrics.

Usage:
    python -m benchmarks.generate_report --metrics benchmark_results/domainmotion/metrics.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_markdown_report(
    metrics_path: Path,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> str:
    """
    Generate a Markdown report from metrics.
    
    Args:
        metrics_path: Path to metrics.json
        output_path: Optional path to save report
        title: Optional report title
        
    Returns:
        Markdown report string
    """
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    if title is None:
        title = f"ConforMix Benchmark Report"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    successful = [m for m in metrics if m.get("error_message") is None]
    failed = [m for m in metrics if m.get("error_message") is not None]
    
    rmsds_to_alt = [m["min_rmsd_to_alt"] for m in successful if m.get("min_rmsd_to_alt") is not None]
    gt_rmsds = [m["ground_truth_rmsd"] for m in successful if m.get("min_rmsd_to_alt") is not None]
    coverages = [m["conformational_coverage"] for m in successful if m.get("conformational_coverage") is not None]
    diversities = [m["rmsd_diversity"] for m in successful if m.get("rmsd_diversity") is not None]
    
    lines = [
        f"# {title}",
        "",
        f"**Generated:** {timestamp}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Proteins | {len(metrics)} |",
        f"| Successful | {len(successful)} ({100*len(successful)/len(metrics):.1f}%) |",
        f"| Failed | {len(failed)} |",
    ]
    
    if rmsds_to_alt:
        lines.append(f"| Avg Min RMSD to Alt | {np.mean(rmsds_to_alt):.2f}Å ± {np.std(rmsds_to_alt):.2f}Å |")
    if gt_rmsds:
        lines.append(f"| Avg Ground Truth RMSD | {np.mean(gt_rmsds):.2f}Å ± {np.std(gt_rmsds):.2f}Å |")
    if coverages:
        lines.append(f"| Avg Conformational Coverage | {np.mean(coverages)*100:.1f}% ± {np.std(coverages)*100:.1f}% |")
    if diversities:
        lines.append(f"| Avg Sample Diversity | {np.mean(diversities):.2f}Å ± {np.std(diversities):.2f}Å |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Per-Protein Results",
        "",
        "| System ID | PDB1 | PDB2 | GT RMSD | Min RMSD to Alt | Coverage | Samples |",
        "|-----------|------|------|---------|-----------------|----------|---------|",
    ])
    
    for m in sorted(metrics, key=lambda x: x.get("min_rmsd_to_alt") or 999):
        system_id = m["system_id"]
        pdb1 = m["pdb1"]
        pdb2 = m["pdb2"]
        gt_rmsd = m.get("ground_truth_rmsd", 0)
        min_rmsd = m.get("min_rmsd_to_alt")
        coverage = m.get("conformational_coverage")
        num_samples = m.get("num_samples", 0)
        error = m.get("error_message")
        
        if error:
            lines.append(f"| {system_id} | {pdb1} | {pdb2} | {gt_rmsd:.1f}Å | [X] Error | - | 0 |")
        else:
            min_rmsd_str = f"{min_rmsd:.2f}Å" if min_rmsd is not None else "-"
            coverage_str = f"{coverage*100:.0f}%" if coverage is not None else "-"
            lines.append(f"| {system_id} | {pdb1} | {pdb2} | {gt_rmsd:.1f}Å | {min_rmsd_str} | {coverage_str} | {num_samples} |")
    
    if failed:
        lines.extend([
            "",
            "---",
            "",
            "## Failed Proteins",
            "",
            "| System ID | Error |",
            "|-----------|-------|",
        ])
        
        for m in failed:
            error = m.get("error_message", "Unknown error")[:80]
            lines.append(f"| {m['system_id']} | {error} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Interpretation Guide",
        "",
        "- **Min RMSD to Alt**: Minimum RMSD from any generated sample to the alternate (target) conformation",
        "- **Coverage**: How close the best sample is to the known alternate state (higher is better)",
        "- **Samples**: Number of conformational samples generated",
        "",
        "> Lower Min RMSD to Alt indicates better conformational sampling.",
        "> High coverage (>50%) suggests ConforMix successfully found the alternate state.",
        "",
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Saved report to {output_path}")
    
    return report


def generate_html_report(
    metrics_path: Path,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> str:
    """
    Generate an HTML report from metrics.
    
    Args:
        metrics_path: Path to metrics.json
        output_path: Optional path to save report
        title: Optional report title
        
    Returns:
        HTML report string
    """
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    if title is None:
        title = "ConforMix Benchmark Report"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    successful = [m for m in metrics if m.get("error_message") is None]
    failed = [m for m in metrics if m.get("error_message") is not None]
    
    rmsds_to_alt = [m["min_rmsd_to_alt"] for m in successful if m.get("min_rmsd_to_alt") is not None]
    coverages = [m["conformational_coverage"] for m in successful if m.get("conformational_coverage") is not None]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .summary-card .label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .success {{ color: #4CAF50; }}
        .error {{ color: #f44336; }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    
    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{len(metrics)}</div>
            <div class="label">Total Proteins</div>
        </div>
        <div class="summary-card">
            <div class="value success">{len(successful)}</div>
            <div class="label">Successful</div>
        </div>
        <div class="summary-card">
            <div class="value error">{len(failed)}</div>
            <div class="label">Failed</div>
        </div>
        <div class="summary-card">
            <div class="value">{np.mean(rmsds_to_alt):.2f}Å</div>
            <div class="label">Avg Min RMSD to Alt</div>
        </div>
        <div class="summary-card">
            <div class="value">{np.mean(coverages)*100:.0f}%</div>
            <div class="label">Avg Coverage</div>
        </div>
    </div>
    
    <h2>Per-Protein Results</h2>
    <table>
        <thead>
            <tr>
                <th>System ID</th>
                <th>PDB1</th>
                <th>PDB2</th>
                <th>GT RMSD</th>
                <th>Min RMSD to Alt</th>
                <th>Coverage</th>
                <th>Samples</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for m in sorted(metrics, key=lambda x: x.get("min_rmsd_to_alt") or 999):
        system_id = m["system_id"]
        pdb1 = m["pdb1"]
        pdb2 = m["pdb2"]
        gt_rmsd = m.get("ground_truth_rmsd", 0)
        min_rmsd = m.get("min_rmsd_to_alt")
        coverage = m.get("conformational_coverage")
        num_samples = m.get("num_samples", 0)
        error = m.get("error_message")
        
        if error:
            html += f"""            <tr class="error">
                <td>{system_id}</td>
                <td>{pdb1}</td>
                <td>{pdb2}</td>
                <td>{gt_rmsd:.1f}Å</td>
                <td>[X] Error</td>
                <td>-</td>
                <td>0</td>
            </tr>
"""
        else:
            min_rmsd_str = f"{min_rmsd:.2f}Å" if min_rmsd is not None else "-"
            coverage_str = f"{coverage*100:.0f}%" if coverage is not None else "-"
            html += f"""            <tr>
                <td>{system_id}</td>
                <td>{pdb1}</td>
                <td>{pdb2}</td>
                <td>{gt_rmsd:.1f}Å</td>
                <td>{min_rmsd_str}</td>
                <td>{coverage_str}</td>
                <td>{num_samples}</td>
            </tr>
"""
    
    html += """        </tbody>
    </table>
</body>
</html>
"""
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"Saved HTML report to {output_path}")
    
    return html


def generate_report(
    metrics_path: Path,
    output_dir: Optional[Path] = None,
    title: Optional[str] = None,
) -> tuple[Path, Path]:
    """
    Generate both Markdown and HTML reports.
    
    Args:
        metrics_path: Path to metrics.json
        output_dir: Directory to save reports
        title: Optional report title
        
    Returns:
        Tuple of (markdown_path, html_path)
    """
    if output_dir is None:
        output_dir = metrics_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    
    generate_markdown_report(metrics_path, md_path, title)
    generate_html_report(metrics_path, html_path, title)
    
    return md_path, html_path


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate reports from ConforMix benchmark metrics"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save reports (default: same as metrics)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Report title"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "both"],
        default="both",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.metrics.parent
    
    if args.format == "markdown":
        md_path = output_dir / "report.md"
        generate_markdown_report(args.metrics, md_path, args.title)
        print(f"Generated: {md_path}")
    elif args.format == "html":
        html_path = output_dir / "report.html"
        generate_html_report(args.metrics, html_path, args.title)
        print(f"Generated: {html_path}")
    else:
        md_path, html_path = generate_report(args.metrics, output_dir, args.title)
        print(f"Generated: {md_path}")
        print(f"Generated: {html_path}")


if __name__ == "__main__":
    main()
