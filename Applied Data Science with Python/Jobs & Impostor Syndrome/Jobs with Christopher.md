# Roles

These are the common **individual contributor (IC)** roles in data science:

- **Data Scientist:** Catch-all term that could mean anything: analyze large datasets to derive actionable insights and build predictive models
- **Data Analyst:** Interpret data and analyze results using statistical techniques to provide reports
- **ML Engineer:** Design and implement machine learning applications and systems
- **Applied Scientist/Researcher:** Conduct scientific research to advance the field of machine learning and data science
- **Analytics Engineer:** Build and manage data pipelines and analytics tools to make data more accessible
- **Data Engineer:** Develop, construct, test, and maintain architectures such as databases and large-scale processing systems

## Levels

- **Junior:** Entry-level position requiring less than 2 years of experience; focuses on learning and growth
- **Intermediate (no modifier):** Mid-level professionals with 2-5 years of experience; capable of independent work and some leadership
- **Senior:** Experienced professionals with over 5 years of experience; lead projects and mentor juniors
- **Tech Lead:** Technical experts who guide development and strategic decisions in projects
- **Staff:** Highly experienced professionals with specialized skills, often involved in strategic planning
- **Principal:** Top-level professionals with significant industry impact, often involved in long-term strategic direction
- **Lead Data Scientist or Head of Data Science:** Varies! Could be a leader of teams or projects, responsible for outcomes; could be a senior contributor with technical leadership responsibilities but no direct reports. Experience level is usually **staff**-level.
- **(Senior) Manager:** Focus on managing direct reports (individual contributors), projects, and deliverables, with a focus on administration and leadership
- **Director:** Senior management role overseeing departments or large groups (possibly including managers), strategic planning, and decision-making

# Interview Steps

1. **Recruiter**
2. **Hiring manager**
3. **Technical**
    1. Screener
    2. Take-home exercise
    3. SQL pair programming
    4. Python pair programming
    5. Case study (analytics, ML, statistics, etc)
4. **Interview panel**
    1. Culture fit
    2. Experience in subject area
    3. Stakeholders

# Technical Questions

- **Explain your thought process:** Describe the steps you would take to solve the problem, including how you would break down the problem into manageable pieces and the reasoning behind your approach
- **Be specific about technologies/methods:** Mention specific programming languages, frameworks, or statistical methods you would use and why
- **Highlight decisions and trade-offs:** Discuss why you chose one approach over another and what you had to compromise
- **Acknowledge limitations:** Be honest about the potential flaws or limitations in your approach and how you might address them
- **You don’t know everything:** It’s okay to admit if you don’t know something, but demonstrate how you would find the answer

# Experience Questions: STAR Method

- **Situation:** Set the scene and give the necessary details of your example
- **Task:** Describe what your responsibility was in that situation
- **Action:** Explain exactly what steps you took to address it
- **Result:** Share what outcomes your actions achieved

### Example question

> Tell me about a time when you encountered unexpected results in your data analysis, and how you addressed it.

### Example response

- **Situation**: During a project to identify risk factors for post-surgical infections, an unexpected correlation between a variable (clinic performing surgery) and infection rates emerged in a Generalized Linear Model (GLM)
- **Task**: Uncover patterns and risk factors for increased infection rates to determine if the clinics are delivering different procedure quality, or if another factor is responsible
- **Action**:
    - Applied hierarchical modeling with random intercepts for clinics to account for patient clustering using Python’s `statsmodels` library
    - Included patient demographics and pathology as covariates in a multivariate regression model in the first level of the model
    - Implemented mixed-effects models to account for the clustering of patients by clinic, allowing for random intercepts for each clinic in the second level of the model
    - Engaged with medical experts and consulted literature to ensure clinical validity of the factors included
- **Result**:
    
    - The analysis revealed significant differences in patient demographics and types of pathology treated at each clinic, which were initially confounding the correlation between clinic and infection rates
    - The refined model, including demographics and pathology severity, explained **85% of the variance** in infection rates
    - Controlling for demographics and pathology severity, **the difference between clinics was no longer significant (p=0.35)**, indicating that the initial observed disparities were due to patient mix rather than quality of care
    
    # Workshop
    
    In small groups, take turns asking and answering the kinds of questions you want more practice on (5-10 minutes per question). If applicable, have the interviewee write code (or pseudocode) on a shared screen. Interviewers should take notes, ask follow-ups, and give feedback on the correctness and completeness of the response.
    
    ## Data Analysis & Interpretation
    
    1. **Given a dataset with customer purchase behavior, how would you identify key factors driving repeat purchases?**
        - Break down the steps you would take to clean and analyze the data.
        - Discuss potential statistical methods or models you might use.
    2. **Imagine you've found a significant drop in website traffic after a recent update. How would you approach diagnosing the issue?**
        - Outline the data you would examine.
        - Describe the analytical techniques to pinpoint the problem areas.
    3. **Analyzing Hospital Readmission Rates**
        - Given a dataset with patient demographics, diagnosis, treatment details, and readmission information, how would you analyze the factors contributing to higher hospital readmission rates? Discuss the types of analyses you would perform and potential challenges you might encounter due to the nature of health data.
        - Consider privacy concerns, data imbalances, and the handling of missing data in your analysis approach.
    4. **Predicting Diabetes Onset**
        - With a dataset containing patient records including age, BMI, insulin levels, and glucose levels, how would you approach predicting the onset of diabetes? Discuss the preprocessing steps, feature selection, and the choice of predictive model.
        - Explore how you would address potential biases in the dataset and evaluate the model's performance, considering the implications of false positives and false negatives in a healthcare context.
    5. **Mental Health in Tech Industry Analysis**
        - Using survey data from tech industry workers on mental health conditions, workplace policies, and utilization of mental health resources, how would you identify key factors that influence the likelihood of employees discussing mental health with their employers? Discuss your approach to analyzing this sensitive data, including any ethical considerations.
        - Consider how you would visualize the relationships between workplace environment and mental health outcomes, and how these insights could inform HR policies.
    6. **Effectiveness of Different Treatment Plans in Chronic Diseases**
        - Given a longitudinal dataset tracking patients with a chronic disease, treatment plans administered, and outcomes over time, how would you analyze the effectiveness of different treatment plans? Discuss the statistical models or methods you would use to account for the longitudinal nature of the data and potential confounding factors.
        - Reflect on how you would handle varying lengths of follow-up and missing data points, and how you would present your findings to both clinical and non-clinical stakeholders.
    7. **COVID-19 Vaccine Efficacy Analysis**
        - With data from clinical trials of different COVID-19 vaccines, including participant demographics, vaccine allocation, and subsequent infection rates, how would you assess and compare the efficacy of the vaccines? Discuss the challenges of working with clinical trial data and how you would ensure the robustness of your findings.
        - Delve into the considerations for subgroup analyses (e.g., by age, pre-existing conditions) and how you would communicate the implications of your analysis for public health policies.
    
    ## Machine Learning
    
    1. **You're tasked with improving an existing recommendation system for an e-commerce platform. How would you proceed?**
        - Discuss how you would evaluate the current system's performance.
        - Propose methods or algorithms to test for potential improvements.
    2. **If you had to predict customer churn, what features would you consider and what model would you use?**
        - Identify the type of data and features that might be relevant.
        - Justify your choice of model and explain how you would validate its performance.
    3. **Predicting Housing Prices**
        - Given a dataset with features like square footage, number of bedrooms, number of bathrooms, and zip code, how would you predict the price of a house? Discuss which type of machine learning model you would use and why.
        - Consider the preprocessing steps needed for this dataset and how you would evaluate the performance of your model.
    4. **Email Spam Detection**
        - You are tasked with designing a spam detection system for email. Discuss what features you might extract from the email text and any metadata, and what kind of machine learning model would be appropriate for this task.
        - Explore the trade-offs between false positives and false negatives in this context and how you would address class imbalance.
    5. **Customer Segmentation for Marketing Strategies**
        - Given customer demographic data and past purchase history, how would you segment customers into distinct groups for targeted marketing campaigns? Discuss the choice of clustering algorithm and how you would determine the number of clusters.
        - Delve into how you would validate the effectiveness of your segmentation and how it could be used in developing marketing strategies.
    6. **Time Series Forecasting for Product Demand**
        - Considering monthly sales data for various products over the past 5 years, how would you forecast demand for the next year? Discuss the time series model(s) you would consider and any seasonality or trends you would expect to account for.
        - Reflect on how you would assess the accuracy of your forecasts and adjust for any anomalies or outliers in the historical data.
    7. **Predicting Customer Churn**
        - With a dataset containing customer usage patterns, service complaints, and demographic information, how would you build a model to predict which customers are most likely to churn? Highlight the steps for feature engineering and model selection.
        - Discuss how you would use the model's predictions to inform retention strategies, and how you might measure the impact of those strategies.
    8. **Sentiment Analysis on Product Reviews**
        - Given a collection of product reviews, each labeled as positive, negative, or neutral, how would you train a model to automatically classify the sentiment of new reviews? Discuss the process of transforming text data into features and the type of model you'd use.
        - Consider how you would handle sarcasm, negations, and context-dependent meanings in the reviews. Discuss how you would evaluate the model's performance in accurately reflecting sentiments.
    
    ## Coding & Algorithms
    
    1. **Write a function to find the nth Fibonacci number. Then, discuss the time complexity of your solution.**
        - Encourage different approaches (e.g., recursive, iterative, memoization).
        - Discuss the trade-offs of each approach in terms of complexity.
    2. **Find the Missing Number in an Array**
        - You're given an array containing `**n**` distinct numbers taken from `**0, 1, 2, ..., n**`. Find the one that is missing from the array.
        - Discuss the approach and how to achieve it with a time complexity of O(n) and space complexity of O(1).
    3. **Determine if a String is an Anagram of Another**
        - Write a function to check if two strings are anagrams of each other (i.e., they contain the same letters in a different order).
        - Consider edge cases and discuss the efficiency of your solution.
    4. **Merge Two Sorted Arrays**
        - Given two sorted arrays, write a function to merge them into a single, sorted array.
        - Explore in-place merging vs. using extra space, discussing the trade-offs.
    
    ## SQL & Data Manipulation
    
    1. **Given tables for orders and products, write an SQL query to find the top 3 most sold products by quantity.**
        - Discuss any JOIN operations or subqueries needed.
        - Explain how you would handle ties for the third place.
    2. **Write an SQL query to identify users with consecutive login days in the past week.**
        - Break down the logic to track consecutive days.
        - Consider the use of window functions or date operations.
    3. **List Employees with Higher Salaries than Their Managers**
        - Given a table `**Employee**` with columns `**Id**`, `**Name**`, `**Salary**`, and `**ManagerId**` where `**ManagerId**` is a foreign key to `**Id**`, write an SQL query to find employees who earn more than their managers.
        - Discuss how self-joins work and their use cases.
    4. **Find the Second Highest Salary**
        - Write an SQL query to find the second highest salary from the `**Employee**` table. If there is no second highest salary, the query should return `**null**`.
        - Explore solutions using subqueries and window functions, and discuss their differences.
    5. **Department Highest Salary**
        - The `**Employee**` table holds all employees. Every employee has an `**Id**`, a `**salary**`, and there is also a column for the `**department Id**`. Write a SQL query to find the employee with the highest salary in each of the departments.
        - Consider using `**JOIN**` with `**GROUP BY**` and window functions. Discuss the performance and readability of your solution.
    6. **Count of Orders Placed on the Same Day by Each Customer**
        - Given an `**Orders**` table with columns `**OrderId**`, `**CustomerId**`, and `**OrderDate**`, write an SQL query to count the number of orders each customer placed on the same day. The output should include the `**CustomerId**`, `**OrderDate**`, and the count of orders.
        - Discuss approaches using `**GROUP BY**` and the significance of date handling in SQL.
    
    ## STAR Method Practice
    
    1. **Describe a project where the initial approach failed and you had to pivot.**
        - Detail the original approach and why it failed.
        - Discuss how you identified a new direction and the outcome.
    2. **Talk about a time when you had to work with a difficult team member to complete a project.**
        - Set the scene with the team dynamics.
        - Explain your approach to collaboration and conflict resolution.