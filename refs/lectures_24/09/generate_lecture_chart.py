#!/usr/bin/env python3
"""
Generate the Altair chart example for Lecture 09.
This creates the actual JSON specification and image that should be shown in the lecture.
"""

import altair as alt
import pandas as pd
from pathlib import Path
import json

# Configure Altair
alt.data_transformers.enable("json")


def main():
    print("🏥 Generating Lecture 09 Altair example...")

    # Create the exact data from the lecture
    physio_df = pd.DataFrame({
        "age": [65, 70, 55, 80, 62, 75, 58, 72],
        "heart_rate": [75, 88, 60, 92, 70, 85, 65, 90],
        "patient_id": [
            "P001",
            "P002",
            "P003",
            "P004",
            "P005",
            "P006",
            "P007",
            "P008",
        ],
        "gender": [
            "Male",
            "Female",
            "Male",
            "Female",
            "Female",
            "Male",
            "Male",
            "Female",
        ],
    })

    # Create the exact chart from the lecture
    scatter_plot = (
        alt.Chart(physio_df)
        .mark_point(size=100)
        .encode(
            x="age:Q",  # Age on x-axis, quantitative
            y="heart_rate:Q",  # Heart rate on y-axis, quantitative
            color="gender:N",  # Color points by gender (nominal)
            tooltip=[
                "patient_id:N",
                "age:Q",
                "heart_rate:Q",
                "gender:N",
            ],  # Info on hover
        )
        .properties(title="Age vs. Heart Rate by Gender")
        .interactive()
    )  # Enable pan and zoom

    # Create media directory if it doesn't exist
    media_dir = Path("media")
    media_dir.mkdir(exist_ok=True)

    # Save as JSON
    json_path = media_dir / "altair_scatter_example.json"
    scatter_plot.save(str(json_path))
    print(f"✅ Saved JSON specification: {json_path}")

    # Save as PNG (requires vl-convert-python)
    try:
        png_path = media_dir / "altair_scatter_plot_python_gen_05.png"
        scatter_plot.save(str(png_path), scale_factor=2)
        print(f"✅ Saved PNG image: {png_path}")
    except Exception as e:
        print(f"⚠️  PNG save failed: {e}")
        print(
            "Install vl-convert-python for PNG export: pip install vl-convert-python"
        )

    # Save as HTML for viewing
    html_path = media_dir / "altair_scatter_example.html"
    scatter_plot.save(str(html_path))
    print(f"✅ Saved HTML file: {html_path}")

    # Read and display the JSON specification
    with open(json_path, "r") as f:
        json_spec = json.load(f)

    print("\n📋 Generated JSON Specification:")
    print("=" * 50)
    print(json.dumps(json_spec, indent=2))

    print(f"\n🎯 Files created in {media_dir}:")
    for file in media_dir.glob("altair_scatter_*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
