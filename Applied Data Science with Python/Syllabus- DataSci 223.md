Faculty: Christopher Seaman

TA: Emily Tang

Dates: January 8 - March 18, 2024 _(no lecture on Jan 15 or Feb 19)_

Lecture: Mondays, 3:10 - 5:00 PM, Mission Hall 1401

Lab: Fridays, 9:10 - 10:30 AM, Mission Hall 1407

# Overview

At the conclusion of this course, students will be able to:

- Develop Proficiency in Python Programming and Data Science Tools: Equip students with the skills to proficiently use essential data science technologies and processes common in industry; including Python, Git, SQL, Pandas, and data visualization libraries, providing a strong foundation for conducting data analysis in real-world scenarios.
- Apply Machine Learning to Solve Real-World Problems: Enable students to apply machine learning techniques including data cleaning, classification and time series analysis, to real-world data science challenges, emphasizing their practical application in fields like health data science.
- Implement Advanced Data Science Concepts: Empower students to implement advanced data science solutions, including generative AI and Large Language Model (LLM) development, with a focus on their relevance and applications in healthcare and health data analysis.
- Develop Effective Data Communication Skills: Enhance students' ability to communicate data insights effectively through data visualization and storytelling, a crucial skill for conveying findings in health data science contexts.

## Prerequisites

Familiarity with programming concepts, including loops, variables, and functions. Ideally, hands-on experience writing and running scripts such as in: Python, R, Bash, or other programming languages.

## Format

Each week will focus on a different area of applied data science. See the list of lecture by week for more details.

- Lectures will be held on Mondays, 3:00 - 5:00 PM
- Labs will be held on Fridays, 9:00 - 10:30 AM

Lectures will provide an overview of new concepts and tools introduced that week. Each lecture will close with a hands-on exercise with solutions due the following lecture.

Labs will not introduce new material; instead, they provide a forum for collaboration between students and staff to help each other with the current material.

Students are encouraged to collaborate in small groups, but may also work independently. The class culminates with a multi-week project applying advanced techniques or combining applications from multiple focus areas.

## Grading

Final grades will be based on the class participation (40%) and submitted exercises (60%).

# Materials

This course will utilize freely available and open-source materials.

## UCSF Course Materials

- Box - Lectures notes/slides and other materials
- CLE Dashboard - Syllabus, discussion, and lecture recordings
- GitHub - Exercises and code, where assignments will be submitted

## Tools

Students should expect to become familiar with:

- Markdown
- Python + data science libraries
- Git + GitHub
- Jupyter notebooks, including Google Colab
- Visual Studio Code

## References

### Recommended

- Python: general introduction
    - _Whirlwind Tour of Python_, VanderPlas - author’s [website](https://jakevdp.github.io/WhirlwindTourOfPython/)
    - _Think Python_, Downey - purchase or read for free at [Green Tea Press](https://greenteapress.com/wp/think-python/)
    - _Hitchhiker’s Guide to Python!_ - official [documentation](https://docs.python-guide.org/)
- Python: data science and analysis
    - _Python for Data Analysis_, McKinney - author’s [website](https://wesmckinney.com/book/)
    - _Python Data Science Handbook,_ VanderPlas - author’s [website](https://jakevdp.github.io/PythonDataScienceHandbook/)

### Optional

- [pythonbooks.org](https://pythonbooks.org/free-books/) - additional free books available
- Python: general introduction
    - _Python 101_ - community resource on the [web](https://python101.pythonlibrary.org/)
    - _Python for Everybody_ - author’s [website](https://www.py4e.com/)
- Python: data science
    - _Machine Learning from Scratch_ - author’s [website](https://dafriedman97.github.io/mlbook/content/introduction.html), learn the nuts and bolts behind the tools
    - _Artificial Intelligence: A Modern Approach_ - author’s [website](https://aima.cs.berkeley.edu/), authoritative deep survey of AI topics
- Python: library-specific
    - _Effective Pandas_ - author’s [website](https://leanpub.com/effective-pandas), deep dive into a common python library not covered in this course
    - _Deep Learning with PyTorch_ - read on publisher’s [website](https://www.manning.com/books/deep-learning-with-pytorch)
    - _PyTorch Tutorials_ - official [documentation](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
    - _TensorFlow Tutorials_ - official [documentation](https://www.tensorflow.org/tutorials)

# Lectures

## Getting Started with Git, Markdown, and Python

- **Version Control using Git:** Fundamentals of version control using Git, emphasizing its role in collaborative data science projects; how version control facilitates teamwork, tracks changes, and ensures project integrity
- **Markdown Basics:** Explore the basics of Markdown, a lightweight markup language, for creating clear and concise documentation
- **Python Environments:** Setting up a Python environment, installation of libraries, and creating virtual environments

## **Pandas for Data Manipulation and Analysis**

- **Loading and Handling Data:** Load data using different formats
- **Handling Large Datasets:** Common challenges that arise when working with big data (larger than will fit in memory)
- **Exploratory Data Analysis:** Exploratory analysis to understand data distributions, patterns, and outliers; extract meaningful insights
- **Data Munging and Cleaning:** Handling missing values, outliers, and ensuring data quality

## SQL Fundamentals: Querying and Manipulating Data

- **Loading and Inspecting Datasets:** Introduction to SQL for loading and inspecting datasets
- **Structure of SQL Statements:** Break down statements including WHERE, ORDER BY, and GROUP BY, to efficiently query and organize data.
- **Joins:** Techniques for integrating information from multiple tables
- **Data Exploration:** Generating summary statistics and visualizations
- **Advanced Concepts:** Common table expressions, stored procedures, and window functions

## Classification Models

- **Learning Paradigms:** Overview of supervised, unsupervised, and reinforcement learning
- **Supervised Learning:** Widely-used supervised classification algorithms, including logistic regression, decision trees, and random forests
- **Unsupervised Learning:** Examination of unsupervised learning, including clustering and dimensionality reduction methods
- **Evaluating Model Performance:** Methods for assessing model performance through cross-validation and appropriate metrics for classification tasks

## Generative AI with Images

- **GANs (Generative Adversarial Networks):** GANs are powerful for generating realistic data by training a generator and a discriminator in a competitive setting with adversarial training.
- **Diffusion Models:** Diffusion models, like those based on the concept of denoising score matching, offer an alternative approach to generative modeling. They focus on the evolution of a probability distribution over time. This adds diversity to the understanding of generative models.
- **VAE’s (Variational Autoencoders):** VAE’s bring in the idea of learning a probabilistic latent representation of data. This contrasts with GAN’s and diffusion models, which do not explicitly model a latent space. VAE’s are useful when you want a structured latent space and want to generate data with specific characteristics.

## Large Language Models

- **Advanced Textual Comprehension:** Large Language Models (e.g., GPT and BERT) excel in comprehending and generating human-like text, showcasing semantic grasp and nuanced language interpretation
- **Architecture:** The transformative role of transformers and attention mechanisms in constructing contextual language understanding, with a brief detour through information geometry and manifolds
- **Training:** Process for building an LLM from scratch
- **Fine-tuning:** Techniques for refining massive general models to suit subject-specific tasks, drawing parallels with approaches used in diffusion models and VAE’s
- **Prompt Engineering:** Effective strategies for crafting prompts and context to effectively extend the LLM’s beyond their training scope (e.g., one-shot and zero-shot learning)
- **Model Chains:** Creating complex agents by chaining LLM’s in sequence, including analysts and personal assistants

## Experimentation & A/B Testing

- **Metrics and Product Analytics:** Key metrics and product analytics relevant to experimentation and A/B testing, including their role in assessing the performance and impact of changes.
- **Research Design:** Developing a robust research design for A/B testing, including the identification of variables, definition of control and experimental groups, and consideration of randomization.
- **Analysis and Results:** Techniques for analyzing A/B test results, interpreting statistical significance, and drawing meaningful conclusions. Presentation of results to stakeholders and decision-makers.
- **Practical Considerations and Pitfalls:** Practical aspects and common challenges in experimentation and A/B testing: potential pitfalls, ethical considerations, and factors that may impact the reliability of results.

## Data Visualization and Communication

- **Principles of Effective Data Visualization:** Foundational principles of impactful data visualization; emphasizing clarity, interpretability, and compelling storytelling.
- **Introduction to Data Visualization Libraries:** Popular data visualization libraries like Matplotlib, Seaborn, and Plotly
- **Interactive Visualizations:** Techniques for creating interactive visualizations on the web, with BI tools, and shared Jupyter notebooks

## Time Series Analysis and Forecasting

- **Understanding Time Series Data:** Unique characteristics of time series data, focusing on temporal patterns and dependencies.
- **Exploratory Analysis:** Extracting meaningful patterns and trends from time series data
- **Forecasting:** Forecasting techniques, including ARIMA and exponential smoothing

## Current Data Science Landscape

- **Recent Trends:** What’s currently happening in the field and how to stay up-to-date.
- **People, Process, and Tools:** Job roles and responsibilities in data science, how systematic and iterative workflows work, and popular tools/technologies used.
- **Deploying to Production:** The crucial steps involved in deploying a model to the cloud, including considerations for multi-model pipelines.
- **The Dreaded** _**Technical Interview**_**:** Get familiar with what’s usually covered, common challenges, and ways to practice.

  

---

# Additional Topics

These may be included per student request or based on availability of experts in industry.

## Python and SQL Patterns

- Introduction to Python programming basics
- Basics of version control using Git
- Fundamentals of SQL: SELECT statement, WHERE clause, ORDER BY, GROUP BY
- Hands-on exercises to integrate Python and SQL for data manipulation

## Data Exploration and Cleaning in Depth

- Loading and inspecting datasets with SQL
- Advanced SQL concepts: JOIN operations, window functions
- Data exploration techniques: summary statistics, visualization
- In-depth coverage of data cleaning methods in SQL

## Feature Engineering and Selection

- Importance of feature engineering in building effective machine learning models
- Techniques for transforming and creating new features from existing data
- Strategies for selecting relevant features and reducing dimensionality
- Hands-on exercises in feature engineering and selection using Python libraries

## Natural Language Processing (NLP)

- Introduction to NLP and its applications in data science
- Pre-processing textual data: tokenization, stemming, lemmatization
- Building and training NLP models for tasks like sentiment analysis and text classification
- Overview of popular NLP libraries (NLTK, spaCy) and pre-trained language models

## Big Data and Distributed Computing

- Understanding the challenges of working with large datasets
- Introduction to distributed computing frameworks (e.g., Apache Spark)
- Hands-on experience with distributed data processing using Python
- Scalable machine learning with big data tools
- Case studies of big data applications in data science

## Ethics in Data Science

- Exploring ethical considerations in data science projects
- Addressing bias and fairness in machine learning models
- Privacy concerns and best practices for handling sensitive data
- Responsible data collection and usage
- Case studies of ethical dilemmas in real-world data science projects

## Deployment and Model Lifecycle

- Importance of deployment in real-world applications
- Understanding the model lifecycle: from development to deployment
- Tools and frameworks for model deployment
- Best practices for maintaining and updating deployed models