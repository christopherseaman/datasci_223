#!/usr/bin/env python3
"""
Report generator library for neonatal feeding study.
Provides utilities for generating markdown reports with sections, charts, and tables.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional


def create_markdown_section(
    heading: str,
    content: Optional[str] = None,
    table_data: Optional[pd.DataFrame] = None,
    chart_reference: Optional[str] = None,
    image_location: Optional[str] = None,
) -> str:
    """
    Create a markdown section with heading and various content types.

    Args:
        heading (str): Section heading
        content (str, optional): Text content
        table_data (pd.DataFrame, optional): Data for table
        chart_reference (str, optional): Path to chart JSON
        image_location (str, optional): Path to image file

    Returns:
        str: Formatted markdown section
    """
    heading_level = 0
    heading_text = heading
    while heading_text.startswith("#"):
        heading_level += 1
        heading_text = heading_text[1:]
    if heading_level == 0:
        heading_level = 2
    section = f"{'#' * heading_level} {heading_text.strip()}\n\n"

    if content is not None:
        section += f"{content}\n\n"

    if table_data is not None:
        if isinstance(table_data, pd.DataFrame):
            section += table_data.to_markdown(index=False)
        elif isinstance(table_data, dict):
            if all(
                isinstance(v, (int, float, str, bool))
                for v in table_data.values()
            ):
                df = pd.DataFrame({
                    "Variable": list(table_data.keys()),
                    "Value": list(table_data.values()),
                })
                section += df.to_markdown(index=False)
            else:
                for key, value in table_data.items():
                    if isinstance(value, (int, float)):
                        section += f"**{key}**: {value:.2f}\n\n"
                    else:
                        section += f"**{key}**: {value}\n\n"
        else:
            section += str(table_data)
        section += "\n\n"

    if chart_reference is not None:
        section += "```vegalite\n"
        section += "{\n"
        section += f'  "schema-url": "{chart_reference}"\n'
        section += "}\n"
        section += "```\n\n"

    if image_location is not None:
        image_name = Path(image_location).name
        section += f"![{image_name}](media/{image_name})\n\n"

    return section


def generate_report(
    sections: List[
        Tuple[
            str,
            Optional[str],
            Optional[pd.DataFrame],
            Optional[str],
            Optional[str],
        ]
    ],
    output_path: Optional[Path] = None,
    title: str = "Analysis Report",
) -> Path:
    """
    Generate a complete markdown report from a list of sections.

    Args:
        sections: List of section tuples (heading, content, table_data, chart_reference, image_location)
        output_path: Where to save the report
        title: Report title

    Returns:
        Path: Path to generated report
    """
    content = f"# {title}\n\n"

    for section in sections:
        content += create_markdown_section(*section)

    if output_path is None:
        output_path = Path("docs") / "generated_report.md"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"📝 Report saved to: {output_path}")
    return output_path


def save_altair_chart_json(chart, output_path_str: str) -> None:
    """Save an Altair chart as JSON."""
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_file))
    print(f"📊 Chart JSON saved to: {output_file}")


def save_json_data(data: Dict[str, Any], output_path_str: str) -> None:
    """Save data as JSON file."""
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"💾 Data saved to: {output_file}")
