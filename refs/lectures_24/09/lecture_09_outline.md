# Lecture 09: Data Visualization, Diagramming, Reporting & Dashboards - Detailed Outline

**Overall Goal:** Equip students with skills to create insightful diagrams (Mermaid), interactive data visualizations (Altair), automated shareable reports (MkDocs), and simple interactive dashboards (Streamlit), applied to health data contexts, culminating in engaging data storytelling examples.

**Target Audience:** Health data science master's students (beginners in programming).
**Lecture Duration:** 90 minutes.
**Format:** Long-form Markdown (this outline will guide `lecture_09.md`).

---

## 0. Introduction (5 minutes)

*   **Hook:** A common challenge in data science is transforming complex health data into clear, actionable insights. Effective communication is as crucial as the analysis itself. Consider the difference between presenting a dense table of numbers versus an interactive chart or a simple dashboard that highlights key findings.
*   **Lecture Objectives:**
    *   Create diagrams as code using Mermaid for clear process documentation.
    *   Craft interactive visualizations with Altair, including building towards a Gapminder-style dynamic chart.
    *   Build automated, shareable reports using MkDocs for disseminating findings.
    *   Develop simple, practical dashboards and then a more dynamic one with Streamlit for data exploration.
    *   Apply these tools to health data scenarios, focusing on principles of effective communication.
*   **Agenda Overview (Mermaid Diagram):**
    ```mermaid
    graph TD
        A[Intro: The Power of Visual Communication] --> B(Diagramming: Mermaid);
        B --> C(Interactive Viz: Altair);
        C --> D(Automated Reports: MkDocs);
        D --> E(Dashboards: Streamlit);
    ```

---

## 1. Diagramming as Code with Mermaid (15 minutes)

### 1.1. Why "Diagrams as Code"?
*   **Concept:** Treating diagrams as source code offers several advantages. These diagrams are defined using text, making them version-controllable with tools like Git, inherently reproducible, and easier to update systematically.
*   **Benefits:** This approach promotes consistency in styling, facilitates straightforward integration into documentation systems (like MkDocs), enhances collaboration through version control, and can improve accessibility.
*   **Contrast:** GUI-based diagramming tools (e.g., PowerPoint, Lucidchart) offer a visual interface but can be challenging for versioning, reproducibility, and programmatic updates.

### 1.2. Introduction to Mermaid
*   **What is Mermaid?** Mermaid is a JavaScript-based diagramming and charting tool that uses Markdown-inspired text definitions to dynamically create and modify diagrams.
*   **Common Diagram Types:** It supports various diagram types, including Flowcharts (for process visualization), Sequence Diagrams (for interaction timelines), Gantt Charts (for project scheduling), Class Diagrams (for software design), and Entity Relationship Diagrams (for database schemas).
*   **Tools:** Mermaid diagrams can be created and viewed using the online Mermaid Live Editor, various VS Code extensions (e.g., "Markdown Preview Mermaid Support"), and are often supported by MkDocs themes or plugins.

### 1.3. Basic Mermaid Syntax & Examples

#### Flowcharts
*   **Concept:** Visualizing processes.
*   **Reference Card: Mermaid Flowchart**
    *   **Declaration:** `graph TD;` / `graph LR;`
    *   **Nodes:** `id[Text]`, `id(Text)`, `id((Text))`, `id{Text}`
    *   **Links:** `A --> B`, `A --- B`, `A -- Text --> B`
*   **Minimal Example (Health Data Analysis Pipeline):**
    ```mermaid
    graph TD;
        A[Load PhysioNet Data] --> B(Data Cleaning & Preprocessing);
        B --> C{Select Analysis Type};
        C -- Descriptive Stats --> D[Generate Summary Tables];
        C -- Predictive Model --> E[Train & Evaluate Model];
        D --> F[Visualize Key Metrics];
        E --> F;
        F --> G[Compile Report/Dashboard];
    ```

### Demo 1: Mermaid Flowchart
*   (Refer to [`lectures/09/demo/01_mermaid_flowchart.md`](lectures/09/demo/01_mermaid_flowchart.md))

---

## 2. Interactive Data Visualization with Altair (25 minutes)

### 2.1. Beyond Static: The Power of Interaction
*   **Why Interactive?** Interactive visualizations allow users to explore data dynamically through features like tooltips, zooming, panning, and selections. This enhances engagement, facilitates the understanding of complex datasets, and enables users to ask their own questions of the data.

### 2.2. Introduction to Altair
*   **What is Altair?** Altair is a declarative statistical visualization library for Python, built on top of Vega-Lite. "Declarative" means you specify *what* you want to visualize, rather than detailing *how* to draw it step-by-step.
*   **Key Principles (Grammar of Graphics):** Altair follows the Grammar of Graphics, where visualizations are built by mapping data columns to visual properties (encodings) of geometric shapes (marks). The core components are:
    *   **Data:** The dataset (typically a Pandas DataFrame).
    *   **Mark:** The geometric object representing data (e.g., points, bars, lines).
    *   **Encoding:** The mapping of data fields to visual channels like x-position, y-position, color, size, shape.
*   **Benefits:** This approach leads to concise code, aesthetically pleasing default styles, and powerful capabilities for creating complex interactive charts.
*   **Comparison (Briefly):** `plotnine` is another Python library based on the Grammar of Graphics (similar to R's `ggplot2`). Both Altair and `plotnine`/`ggplot2` are declarative, contrasting with the more imperative (step-by-step) approach of basic `matplotlib`.

### 2.3. Basic Altair: Building Blocks
*   **Reference Card: `altair.Chart`**
    *   `alt.Chart(data)`
    *   `.mark_type()` (e.g., `mark_point()`, `mark_bar()`)
    *   `.encode(x='col1:Q', y='col2:N', ...)` (Q: Quant, N: Nominal, O: Ordinal, T: Temporal)
    *   `.properties(width=W, height=H, title='Title')`
    *   `.interactive()`
    *   `.save('filename.html'/'filename.json'/'filename.png')` (mention `altair_viewer`, `vl-convert`)
*   **Minimal Example (Scatter Plot from PhysioNet data snippet):**
    ```python
    import altair as alt
    import pandas as pd
    # Assume physio_df is a Pandas DataFrame loaded from a PhysioNet source
    # with columns like 'age', 'heart_rate', 'patient_id'
    # physio_df = pd.DataFrame({
    #    'age': [65, 70, 55, 80], 'heart_rate': [75, 88, 60, 92], 
    #    'patient_id': ['P001', 'P002', 'P003', 'P004']
    # })
    
    scatter_plot = alt.Chart(physio_df).mark_point(size=100).encode(
        x='age:Q',
        y='heart_rate:Q',
        tooltip=['patient_id:N', 'age:Q', 'heart_rate:Q']
    ).properties(
        title='Age vs. Heart Rate'
    ).interactive()
    # scatter_plot.save('hr_scatter.html')
    # scatter_plot.save('hr_scatter.json') # For MkDocs/Streamlit
    ```

### 2.4. Towards Gapminder: Selections & Layering for Dynamic Charts
*   **Selections:** `alt.selection_interval()`, `alt.selection_point()`, `alt.selection_single()` (for dropdowns/radio buttons via binding).
*   **Input Binding:** `alt.binding_range()` (for sliders), `alt.binding_select()` (for dropdowns).
*   **Conditional Encodings:** `alt.condition(selection, true_value, false_value)`.
*   **Transformations:** `transform_filter()`, `transform_aggregate()`.
*   **Layering & Concatenation:** `chart1 + chart2` (layer), `chart1 | chart2` (horizontal), `chart1 & chart2` (vertical).

**Building Blocks for Dynamic Charts (e.g., for Gapminder-style visualization):**
*   This section focuses on Altair techniques for creating components that can be assembled into highly interactive visualizations, such as a Gapminder-style chart (which will be fully demonstrated with Streamlit later).
*   Key Altair features for such charts:
    *   Encoding for X (e.g., health expenditure, potentially log-scaled), Y (e.g., life expectancy), Size (e.g., population), Color (e.g., region).
    *   Using `alt.selection_single` with `bind=alt.binding_range` to create a time-based slider.
    *   Applying `transform_filter` to link the slider selection to the displayed data.
    *   Ensuring appropriate scales and tooltips for rich interaction.
```python
# Conceptual Altair components for a dynamic time-series scatter plot:
# import altair as alt
# import pandas as pd

# # Assume health_data_df exists with columns:
# # 'country', 'year', 'health_exp_pc', 'life_expectancy', 'population', 'region'

# year_slider = alt.selection_single(
#     name="Select_Year", fields=['year'],
#     bind=alt.binding_range(min=health_data_df['year'].min(), max=health_data_df['year'].max(), step=1),
#     init={'year': health_data_df['year'].min()}
# )

# base_chart = alt.Chart(health_data_df).mark_circle().encode(
#     x=alt.X('health_exp_pc:Q', scale=alt.Scale(type="log")),
#     y=alt.Y('life_expectancy:Q', scale=alt.Scale(zero=False)),
#     size='population:Q',
#     color='region:N',
#     tooltip=['country:N', 'year:O', 'life_expectancy:Q', 'health_exp_pc:Q']
# ).add_params(
#     year_slider
# ).transform_filter(
#     year_slider
# ).properties(
#     width=600, height=400
# )
# # base_chart.save('dynamic_scatter_components.json')
```

### Demo 2: Interactive Altair Chart
*   (Refer to [`lectures/09/demo/02_altair_interactive_chart.md`](lectures/09/demo/02_altair_interactive_chart.md))

---

## 3. Automated Report Generation with MkDocs (20 minutes)

### 3.1. Why Static Site Generators for Reports?
*   **Concept & Benefits:** Static site generators (SSGs) like MkDocs take source files (e.g., Markdown text, images, chart specifications) and templates to produce a complete, self-contained HTML website. For data science reports, this offers shareability (simple HTML files), version control (source files managed with Git), reproducibility (reports can be rebuilt from source), professional appearance through themes, and automation of the report generation process.
*   **MkDocs:** MkDocs is known for its speed, simplicity, and focus on creating project documentation, which extends well to generating data analysis reports.

### 3.2. Setting up MkDocs
*   **Installation:** `pip install mkdocs mkdocs-material mkdocs-altair-plugin pandas altair`
*   **Project Init:** `mkdocs new my_health_report && cd my_health_report`
*   **Directory Structure:** `mkdocs.yml`, `docs/` (index.md, charts/).

### 3.3. Configuring `mkdocs.yml`
*   **Basic:** `site_name`, `theme: material`.
*   **Plugins:** `search`, `altair` (with `vega_lite_version`).
*   **Navigation:** `nav` for structuring the report.

### 3.4. Creating Report Content & Embedding Charts/Diagrams
*   **Markdown:** Narrative in `.md` files in `docs/`.
*   **Python Script for Charts:** (e.g., `scripts/generate_charts.py`) to save Altair charts as JSON into `docs/charts/`.
*   **Embedding Altair Charts:** `{% include_altair "charts/my_chart.json" %}` in Markdown.
*   **Embedding Mermaid Diagrams:** Directly in Markdown using ```mermaid ... ``` blocks (if theme supports it, like Material for MkDocs) or via plugins if needed.

### 3.5. Building, Serving, and Deploying
*   **Build:** `mkdocs build` (generates `site/`).
*   **Serve Locally:** `mkdocs serve` (live reload).
*   **Deploying with GitHub Pages (Conceptual Overview):**
    *   Push `mkdocs.yml` and `docs/` to GitHub.
    *   Use `mkdocs gh-deploy` or GitHub Actions to build and deploy to `gh-pages` branch.

### Demo 3: Automated Report with MkDocs
*   (Refer to [`lectures/09/demo/03_mkdocs_automated_report.md`](lectures/09/demo/03_mkdocs_automated_report.md))
*   This demo will include generating an Altair chart, saving it as JSON, embedding it in an MkDocs page, and also embedding a Mermaid diagram (e.g., a workflow for the data used in the report).

---

## 4. Interactive Dashboards with Streamlit (25 minutes)

### 4.1. From Reports to Interactive Applications
*   **What is Streamlit?** Streamlit is a Python library designed to quickly turn data scripts into shareable web applications. It's often described as "the fastest way to build and share data apps."
*   **Why Streamlit?** It enables rapid prototyping of interactive data tools, requires no prior knowledge of HTML, CSS, or JavaScript for creating simple applications, and maintains a Python-centric workflow.

### 4.2. Streamlit Fundamentals
*   **Installation:** `pip install streamlit altair pandas`
*   **Running:** `streamlit run your_script.py`
*   **Core Concepts:**
    *   Script reruns on interaction.
    *   Layout: `st.title()`, `st.header()`, `st.write()`, `st.markdown()`, `st.sidebar`.
    *   Displaying Data: `st.dataframe()`, `st.table()`, `st.metric()`, `st.json()`.
    *   Charts: `st.altair_chart()`, `st.pyplot()`, `st.plotly_chart()`.
    *   Input Widgets: `st.button()`, `st.slider()`, `st.selectbox()`, `st.multiselect()`, `st.text_input()`, `st.date_input()`. State is handled by Streamlit.

### 4.3. Demo 4: Simple, Practical Health Dashboard with Streamlit
*   (Refer to [`lectures/09/demo/04_streamlit_simple_dashboard.md`](lectures/09/demo/04_streamlit_simple_dashboard.md))
*   **Goal:** Demonstrate immediate utility with a common task using a PhysioNet dataset snippet.
*   **Key features to showcase:**
    *   Loading data (using `@st.cache_data` for efficiency).
    *   Using `st.sidebar` for filters (e.g., `st.selectbox`, `st.slider`).
    *   Filtering a Pandas DataFrame based on widget selections.
    *   Displaying the filtered DataFrame (`st.dataframe`).
    *   Displaying a simple Altair chart based on the filtered data (`st.altair_chart`).
    *   Using `st.metric` to show a key summary statistic.

### 4.4. Demo 5: Mini-Gapminder Style Dashboard with Streamlit
*   (Refer to [`lectures/09/demo/05_streamlit_gapminder_dashboard.md`](lectures/09/demo/05_streamlit_gapminder_dashboard.md))
*   **Goal:** Showcase more dynamic interactivity and data storytelling, building on Altair components.
*   **Key features to showcase:**
    *   Using an appropriate dataset (public Gapminder or a suitable health equivalent).
    *   Implementing a `st.slider` for "Year" selection.
    *   Dynamically filtering the DataFrame passed to an Altair chart based on the slider.
    *   The Altair chart itself will be the multi-encoded scatter plot (X, Y, Size, Color) discussed in section 2.4.
    *   Displaying the interactive Altair chart using `st.altair_chart`.

### 4.5. Sharing Streamlit Apps (Briefly)
*   Streamlit Community Cloud (free sharing).
*   Other deployment options (Docker, cloud platforms) – conceptual mention.

---
**End of Lecture Outline**