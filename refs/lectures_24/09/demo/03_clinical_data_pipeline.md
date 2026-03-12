# Demo 3: Clinical Data Analysis Pipeline

**Goal:** Learn how to build and run an automated clinical research pipeline that generates interactive reports and deploys them to GitHub Pages. This demo uses a real-world inspired neonatal feeding study to teach data pipeline concepts.

## What We're Building

We're exploring a complete clinical research pipeline that studies optimal feeding times for premature babies. The pipeline automatically:

1. **Generates realistic clinical data** based on actual neonatal research patterns
2. **Performs statistical analysis** (correlations, regression, t-tests)
3. **Creates interactive visualizations** using Altair/Vega-Lite
4. **Builds automated reports** with MkDocs
5. **Deploys to GitHub Pages** using GitHub Actions

**Real-World Context:** This pipeline studies optimal feeding times for premature babies - a critical clinical question that affects thousands of infants worldwide.

## Project Structure

All files for this demo are located in the [`mkdocs_report_project`](mkdocs_report_project:1) directory:

```
mkdocs_report_project/
├── .github/workflows/
│   └── deploy.yml              # GitHub Actions for automated deployment
├── docs/
│   ├── charts/                 # Generated interactive charts (JSON)
│   ├── data/                   # Generated datasets and statistics
│   ├── analysis/               # Analysis documentation pages
│   ├── index.md               # Main report page
│   ├── about.md               # Project information
│   ├── interactive_charts.md  # Chart gallery
│   └── methodology.md         # Research methodology
├── reports/
│   └── report_generator.py    # Core analysis and visualization code
├── utils/
│   └── reporting.py           # Utility functions for saving charts
├── orchestrator.py            # Main pipeline script
├── mkdocs.yml                 # Site configuration
└── requirements.txt           # Python dependencies
```

## Step 1: Explore the Enhanced Pipeline

Navigate to the project directory:

```bash
cd lectures/09/demo/mkdocs_report_project
```

### Key Improvements Made

The pipeline has been enhanced with:

- **Realistic neonatal data generation** based on clinical research patterns
- **Comprehensive statistical analysis** including correlations and regression
- **Interactive visualizations** showing gestational age vs feeding outcomes
- **Educational documentation** explaining clinical significance
- **Automated deployment** with GitHub Actions

## Step 2: Examine the Data Generation

Look at [`reports/report_generator.py`](mkdocs_report_project/reports/report_generator.py:1) to see how realistic clinical data is created:

**Key Features:**
- **Gestational age** (24-32 weeks) with realistic distribution
- **Birth weight** correlated with gestational age (~100g per week)
- **Clinical factors** (mechanical ventilation, multiple births)
- **Feeding outcomes** with realistic relationships to patient characteristics

**Clinical Relationships Modeled:**
- Earlier gestational age → longer time to full oral feeding
- Lower birth weight → delayed feeding progression
- Mechanical ventilation → additional feeding delays
- Sex differences in feeding readiness

## Step 3: Understanding the Analysis

The pipeline performs sophisticated analysis:

```python
# Example: Statistical analysis performed
def perform_statistical_analysis(df):
    # Correlation analysis
    correlations = {
        'ga_vs_time_to_fof': df['gestational_age_weeks'].corr(df['time_to_FOF']),
        'weight_vs_time_to_fof': df['birth_weight_grams'].corr(df['time_to_FOF'])
    }
    
    # T-tests for clinical factors
    vent_yes = df[df['mechanical_ventilation'] == 1]['time_to_FOF']
    vent_no = df[df['mechanical_ventilation'] == 0]['time_to_FOF']
    vent_ttest = stats.ttest_ind(vent_yes, vent_no)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['gestational_age_weeks'], df['time_to_FOF']
    )
```

## Step 4: Run the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python orchestrator.py
```

This will:
1. Generate 180 realistic patient records
2. Perform statistical analysis
3. Create 5 interactive charts
4. Build the MkDocs site
5. Display key findings

**Expected Output:**
```
🏥 Neonatal Feeding Study Pipeline
==================================================
📋 Research Question: What factors predict time to full oral feeding
   in premature infants?

📊 Step 1: Generating neonatal feeding study data and analysis...
   📊 Generating realistic neonatal feeding data...
   ✅ Generated 180 patients with realistic clinical relationships
   📈 Performing statistical analysis...
   📊 Generating and saving interactive chart specifications...
   ✅ Generated demographics_overview chart
   ✅ Generated primary_analysis chart
   ✅ Generated feeding_progression chart
   ✅ Generated intervention_timing chart
   ✅ Generated clinical_factors chart

📊 Key Findings:
   • Analyzed 180 premature infants
   • Mean gestational age: 28.5 weeks
   • Mean time to full oral feeding: 15.2 days
   • Primary finding: Each additional week of gestational age reduces time to FOF by 2.5 days
   • Correlation (GA vs Time to FOF): -0.78
```

## Step 5: View the Interactive Report

Serve the documentation site:

```bash
mkdocs serve
```

Open your browser to `http://127.0.0.1:8000` to explore:

- **Interactive scatter plots** with zoom/pan capabilities
- **Hover tooltips** showing patient details
- **Statistical results** embedded in the narrative
- **Clinical insights** and recommendations
- **Professional documentation** layout

## Step 6: Deploy to GitHub Pages

### Set Up Repository

1. **Create a new GitHub repository**
2. **Copy the mkdocs_report_project contents** to your repository
3. **Push to GitHub**:

```bash
git init
git add .
git commit -m "Initial neonatal feeding study pipeline"
git branch -M main
git remote add origin https://github.com/yourusername/neonatal-feeding-study.git
git push -u origin main
```

### Enable GitHub Pages

1. Go to repository **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **gh-pages** (will be created automatically)

### Automatic Deployment

The [`deploy.yml`](mkdocs_report_project/.github/workflows/deploy.yml:1) workflow will:
- **Trigger on every push** to main branch
- **Run the analysis pipeline** automatically
- **Generate fresh data and charts**
- **Build and deploy** the updated site
- **Comment on pull requests** with pipeline results

## Step 7: Understanding the Educational Value

This pipeline demonstrates key concepts:

### Data Science Skills
- **Realistic data generation** with proper statistical relationships
- **Exploratory data analysis** with correlation and regression
- **Hypothesis testing** using t-tests for group comparisons
- **Effect size calculation** and clinical significance interpretation

### Visualization Skills
- **Interactive charts** with Altair/Vega-Lite
- **Publication-quality** graphics with proper labeling
- **Multi-dimensional encoding** (color, size, position)
- **User interaction** (zoom, pan, hover, selection)

### Software Engineering Skills
- **Modular code organization** with clear separation of concerns
- **Automated pipelines** with orchestration scripts
- **Documentation generation** with dynamic content
- **Version control** and collaborative development
- **Continuous deployment** with GitHub Actions

### Clinical Research Skills
- **Research question formulation** and hypothesis testing
- **Study design** with appropriate inclusion/exclusion criteria
- **Statistical analysis** with clinical interpretation
- **Results presentation** for clinical audiences

## Success Validation

✅ **Pipeline Execution**: `python orchestrator.py` runs without errors  
✅ **Data Generation**: Realistic clinical data with proper relationships  
✅ **Statistical Analysis**: Correlations, t-tests, and regression results  
✅ **Chart Creation**: 5 interactive JSON charts generated  
✅ **Site Building**: MkDocs builds professional documentation  
✅ **Local Serving**: `mkdocs serve` displays interactive report  
✅ **GitHub Deployment**: Automatic updates on every push  

## Real-World Applications

This pipeline pattern is used in:

- **Clinical Research**: Automated analysis of patient outcomes
- **Quality Improvement**: Hospital performance monitoring
- **Regulatory Reporting**: Standardized analysis for FDA submissions
- **Academic Research**: Reproducible analysis workflows
- **Healthcare Analytics**: Population health trend analysis

## Key Learning Outcomes

🎯 **Pipeline Architecture**: Understanding automated analysis workflows  
🎯 **Clinical Data Science**: Working with realistic medical data  
🎯 **Interactive Visualization**: Creating publication-quality charts  
🎯 **Report Automation**: Dynamic content generation from data  
🎯 **DevOps for Research**: Automated deployment and version control  
🎯 **Statistical Interpretation**: Translating analysis to clinical insights  

## Next Steps

**Extend the Pipeline:**
1. Add more sophisticated statistical models (logistic regression, survival analysis)
2. Include additional clinical variables (medications, procedures)
3. Create longitudinal analysis tracking patients over time
4. Add machine learning models for outcome prediction
5. Integrate with real electronic health record data

**Improve Visualizations:**
1. Add more interactive features (brushing, linking)
2. Create animated charts showing progression over time
3. Build dashboard-style layouts with multiple linked charts
4. Add geographic analysis if multi-site data available

**Enhance Deployment:**
1. Add automated testing for data quality
2. Include performance monitoring and alerts
3. Create staging environments for testing changes
4. Add user authentication for sensitive data

This demo provides a complete foundation for building professional clinical research pipelines that can scale to real-world applications.