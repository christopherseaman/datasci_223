# Interactive Clinical Research Reports with MkDocs and Altair

This project demonstrates how to create automated, interactive clinical research reports and publish them to GitHub Pages. It's designed for health data science students who are new to programming and want to learn modern documentation practices.

## 🎯 Learning Objectives

By the end of this demo, you will be able to:
- Generate automated statistical reports from clinical data
- Create interactive data visualizations
- Build professional documentation sites
- Deploy reports to GitHub Pages
- Understand basic version control practices

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- Git installed on your computer
- A GitHub account

### 2. Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/neonatal-feeding-study.git
cd neonatal-feeding-study
```

2. **Create and activate a virtual environment:**
```bash
# On macOS/Linux:
python -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

### Scripts
```
.
├── orchestrator.py              # Main pipeline coordinator
├── utils/
│   ├── data_generator.py       # Synthetic data generation
│   ├── report_generator.py     # Report generation utilities
│   ├── visualization.py        # Interactive chart creation
│   └── reports/
│       ├── rpt_10_interactive_visualizations.py  # Interactive viz report
│       └── bivariate_analysis.py                 # Statistical analysis report
└── mkdocs.yml                  # Documentation site configuration
```

### Generated Results
```
results/
├── index.md                    # Landing page (copied from README.md)
├── reports/                    # Analysis reports
│   ├── media/                  # Interactive visualizations
│   │   ├── demographics_overview.json
│   │   ├── primary_analysis.json
│   │   ├── clinical_factors.json
│   │   ├── correlation_matrix.json
│   │   └── ventilation_box_plot.json
│   ├── visualization.md       # Interactive charts report
│   └── bivariate_analysis.md  # Statistical analysis report
└── methodology.md             # Study methodology documentation
```

### Key Components

1. **Data Generation** (`data_generator.py`)
   - Creates synthetic neonatal feeding study data
   - Simulates realistic clinical patterns
   - Generates data dictionary

2. **Visualization** (`visualization.py`)
   - Creates interactive Altair/Vega-Lite charts
   - Handles demographics, primary analysis, and clinical factors
   - Generates correlation matrices and box plots
   - Saves charts as JSON for web embedding

3. **Report Generation** (`report_generator.py`)
   - Generates markdown reports
   - Handles statistical analysis
   - Creates formatted tables and sections

4. **Analysis Reports** (`reports/`)
   - `rpt_10_interactive_visualizations.py`: Interactive charts and exploratory analysis
   - `bivariate_analysis.py`: Statistical relationships and hypothesis testing

## 📊 Running the Pipeline

### 1. Generate Reports

The pipeline will:
- Generate sample neonatal feeding study data
- Perform statistical analysis
- Create interactive visualizations
- Generate both exploratory and statistical reports
- Build the documentation site

```bash
python orchestrator.py
```

### 2. View Your Report Locally

```bash
mkdocs serve
```
Then open your browser to: http://127.0.0.1:8000

### 3. Deploy to GitHub Pages

1. Create a new GitHub repository
2. Update `mkdocs.yml` with your repository details:
   ```yaml
   site_url: 'https://yourusername.github.io/neonatal-feeding-study'
   repo_name: 'yourusername/neonatal-feeding-study'
   repo_url: 'https://github.com/yourusername/neonatal-feeding-study'
   ```
3. Push your changes:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```
4. Enable GitHub Pages in your repository settings

## 🎓 Learning Resources

### Key Concepts

1. **MkDocs**
   - A static site generator for project documentation
   - Uses Markdown for content
   - Supports themes and plugins

2. **Altair/Vega-Lite**
   - Declarative visualization library
   - Creates interactive charts
   - JSON-based specification
   - Supports statistical visualizations

3. **GitHub Pages**
   - Free hosting for static websites
   - Automatic deployment via GitHub Actions
   - Custom domain support

### Common Tasks

1. **Adding a New Chart**
   ```python
   import altair as alt
   
   # Basic scatter plot
   chart = alt.Chart(data).mark_point().encode(
       x='gestational_age',
       y='time_to_fof',
       color='ventilation_status'
   )
   
   # Statistical visualization
   box_plot = alt.Chart(data).mark_boxplot().encode(
       x='ventilation_status',
       y='time_to_fof'
   )
   ```

2. **Creating a New Report Section**
   ```markdown
   # New Analysis Section
   
   ## Methods
   Describe your statistical methods here...
   
   ## Results
   Present your findings here...
   
   ```vegalite
   {
     "schema-url": "media/your_chart.json"
   }
   ```
   ```

## 🔍 Troubleshooting

### Common Issues

1. **MkDocs Build Fails**
   - Check that all dependencies are installed
   - Verify your `mkdocs.yml` configuration
   - Ensure all chart JSON files are valid
   - Check that media paths are correct

2. **Charts Not Displaying**
   - Check browser console for errors
   - Verify Vega-Lite JSON syntax
   - Ensure data files are in correct location
   - Verify media directory structure
   - Make sure Vega-Lite blocks use proper fence syntax:
     ```markdown
     ```vegalite
     {
       "schema-url": "media/chart.json"
     }
     ```
     ```
   - Check that all required JavaScript is loaded:
     ```yaml
     extra_javascript:
       - https://cdn.jsdelivr.net/npm/vega@5
       - https://cdn.jsdelivr.net/npm/vega-lite@5
       - https://cdn.jsdelivr.net/npm/vega-embed@6
     ```
   - If you see "Your vegalite syntax is not valid JSON" error:
     - Make sure the JSON is not double-wrapped in quotes
     - Use single-line JSON format: `{"schema-url": "media/chart.json"}`
     - Avoid extra whitespace or formatting in the JSON

3. **GitHub Pages Not Updating**
   - Check GitHub Actions workflow status
   - Verify repository settings
   - Ensure `mkdocs.yml` has correct site URL
   - Check that all files are committed and pushed

4. **Report Generation Issues**
   - Verify that the `results` directory exists and is writable
   - Check that the `reports` and `reports/media` directories are created
   - Ensure all required Python packages are installed
   - Check that data files are in the correct format

5. **Path Issues**
   - Make sure media paths in reports are relative to the report location
   - Verify that chart JSON files are in the correct media directory
   - Check that mkdocs.yml navigation paths match the actual file structure
   - Ensure all paths use forward slashes (/) even on Windows

6. **Theme and Styling**
   - Verify that the material theme is properly installed
   - Check that custom CSS/JS is loaded correctly
   - Ensure dark/light mode toggle is working
   - Test responsive design on different screen sizes

## 📚 Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Altair Documentation](https://altair-viz.github.io/)
- [GitHub Pages Guide](https://pages.github.com/)
- [Markdown Guide](https://www.markdownguide.org/)

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 