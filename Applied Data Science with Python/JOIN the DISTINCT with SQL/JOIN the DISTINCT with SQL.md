# 💽 Why SQL?

## Data is big

SQL (Structured Query Language) is a powerful tool in the data scientist's arsenal, offering a structured and efficient way to interact with databases. It serves as a standard language for managing and querying relational databases, providing a systematic approach to handle vast datasets. Here are some compelling reasons why SQL is a must-have skill for data scientists.

## **Simplified Data Retrieval**

SQL allows you to retrieve specific data from large datasets with ease. Its simple syntax enables you to express complex queries succinctly, making it a valuable tool for extracting meaningful insights from databases.

## **Standardized Language**

SQL is a standardized language used across different database management systems (DBMS). Whether you're working with MySQL, PostgreSQL, SQLite, or others, the fundamental SQL principles remain consistent. This standardization enhances portability and ensures your skills are transferable across various platforms.

## **Efficient Data Manipulation**

With SQL, you can perform various data manipulation tasks, including filtering, sorting, and aggregating data. It provides a robust framework for handling data at scale, making it an essential skill for anyone dealing with large datasets in a professional setting.

## **Seamless Integration with Python**

The integration of SQL with Python opens up new possibilities for data scientists. You can leverage the strengths of both SQL and Python by combining SQL's data manipulation capabilities with Python's extensive libraries for analysis, visualization, and machine learning.

# 🏗️ Structure of a SQL statement

```SQL
-- Retrieve a list of unique department names and the total number of employees in each department
SELECT DISTINCT department_name, COUNT(employee_id) AS total_employees
FROM employees
JOIN departments ON employees.department_id = departments.department_id
WHERE employees.salary > 50000 -- Filter employees with a salary greater than 50000
GROUP BY department_name
ORDER BY total_employees DESC -- Order the results by total employees in descending order
LIMIT 10; -- Limit the output to the top 10 departments
```

Explanation of each part:

- `**SELECT DISTINCT**`**:** Selects unique department names.
- `**COUNT(employee_id) AS total_employees**`**:** Counts the number of employees in each department and renames the column as "total_employees."
- `**FROM employees**`**:** Specifies the main table as "employees."
- `**JOIN departments ON employees.department_id = departments.department_id**`**:** Joins the "employees" table with the "departments" table based on the department_id.
- `**WHERE employees.salary > 50000**`**:** Filters out employees with a salary less than or equal to 50000.
- `**GROUP BY department_name**`**:** Groups the results by department_name.
- `**ORDER BY total_employees DESC**`**:** Orders the results by the total number of employees in descending order.
- `**LIMIT 10**`**:** Limits the output to the top 10 departments.

## Read SQL the way a computer does

Start with the `select`, then walk through the execution order

![[reading_sql.png]]

# 🤿 Deep-dive

Now, let's take a deep dive into crafting SQL queries. SQL queries are used to retrieve specific information from databases. Here are some essential query types:

## Semicolons `;` and comments `--`

SQL statements are separated using semicolons. You may send multiple statements at the same time as long as they are separated by semicolons. **NOTE**: your environment may expect to receive only a single table as a response, so multiple `select` statements may NOT be valid.

Unlike python, SQL uses double-dashes to indicate comments. Comments may be on their own lines or on the same line as code.

## `**SELECT**`

The SELECT statement is fundamental for retrieving data. It allows you to specify the columns you want to retrieve and the conditions for selecting rows. For example:

```SQL
SELECT column1, column2
FROM table
WHERE condition;
```

The `**SELECT**` statement is versatile and can be customized to fetch specific columns from a table based on specified conditions.

## `**UPDATE**`**,** `**INSERT**`**,** `**DELETE**`

These statements are used to modify data in the database:

- `**UPDATE**`**:** Modifies existing records in a table based on specified conditions.
    
    ```SQL
    UPDATE table
    SET column1 = value1
    WHERE condition;
    ```
    
- `**INSERT**`**:** Adds new records to a table.
    
    ```SQL
    INSERT INTO table (column1, column2)
    VALUES (value1, value2);
    ```
    
- `**DELETE**`**:** Removes records from a table based on specified conditions.
    
    ```SQL
    DELETE FROM table
    WHERE condition;
    ```
    

These statements are powerful but should be used with caution, especially `**DELETE**`, to avoid unintended data loss.

## `**JOIN**` **operations**

JOIN operations are crucial for combining data from multiple tables. They enable you to correlate information based on common columns.

```SQL
SELECT *
FROM table1
INNER JOIN table2 ON table1.column1 = table2.column2;
```

You may also use `**USING(column)**` instead of `**ON**` when table1 and table2 have a column of the same name and type being used for the join.

```SQL
-- Sample employees table
CREATE TABLE employees (
  employee_id INT PRIMARY KEY,
  first_name VARCHAR(50),
  last_name VARCHAR(50),
  department_id INT
);

-- Sample departments table
CREATE TABLE departments (
  department_id INT PRIMARY KEY,
  department_name VARCHAR(50)
);

-- Joining employees and departments and renaming columns
SELECT 
  e.employee_id,
  e.first_name,
  e.last_name,
  e.department_id AS employee_department_id,  -- Renaming employee department_id
  d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;
```

### **Types of** `**JOIN**`**'s**

Common types include INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL JOIN.

- `**INNER JOIN**`**:** Returns rows where there is a match in both tables.
- `**LEFT JOIN**`**:** Returns all rows from the left table and matched rows from the right table.
- `**RIGHT JOIN**`**:** Returns all rows from the right table and matched rows from the left table.
- `**FULL JOIN**`**:** Returns all rows when there is a match in either the left or right table.

![[join_types.png]]

### **The Perils of** `**FULL JOIN**`**: Avoiding Execution Explosion**

> [!important]  
> Unspecified JOIN will default for a FULL JOIN!!!  

When the join type is not specified, SQL defaults to a FULL JOIN. This means that all rows from both tables are combined, matching where possible and filling in NULLs for non-matching rows.

While FULL JOINs can be useful in specific scenarios, they come with significant risks, especially in terms of performance and result set size.

1. **Cartesian Product:**
    
    > FULL JOINs can lead to a Cartesian product, where each row from the first table is combined with every row from the second table. This results in a potentially massive result set.
    
2. **Data Explosion:**
    
    > The number of rows in the result set of a FULL JOIN can be much larger than expected, especially when dealing with tables of substantial size.
    
3. **Performance Impact:**
    
    > FULL JOINs often have a significant impact on query performance due to the need to process and combine all rows from both tables. This can lead to longer execution times and increased resource consumption.
    
4. **Resource Intensive:**
    
    > The operation of a FULL JOIN requires more resources compared to other types of joins. It can strain the database server, affecting the overall performance of the system.
    
5. **Unexpected Results:**
    
    > The expansive result set generated by a FULL JOIN may contain unexpected combinations of data, leading to difficulties in interpreting and utilizing the results.
    

### **Best Practice:**

- **Explicitly Specify Join Types:**
    
    > Always explicitly specify the type of join you intend to use (e.g., INNER JOIN, LEFT JOIN). This helps avoid the pitfalls of unintentional FULL JOINs.
    
- **Understand Data Relationships:**
    
    > Clearly understand the relationships between tables and choose the appropriate join type based on the desired outcome of the query.
    
      
    

## Filtering `WHERE` you want

The `**WHERE**` clause serves as your data gatekeeper, allowing you to filter rows based on specific conditions. It operates before grouping and aggregation, helping you focus on the data that truly matters.

**Example:**

```SQL
-- Selecting employees with a salary greater than $50,000
SELECT *
FROM employees
WHERE salary > 50000;
```

## `**GROUP BY**` **and Aggregate Harmony:**

The synergy between `**GROUP BY**` and aggregate functions lets you perform calculations on subsets of data. It's a dynamic duo that brings order to the chaos, grouping rows and providing valuable insights.

**Example:**

```SQL
-- Calculating the total sales for each product category
SELECT category, SUM(sales) AS total_sales
FROM products
GROUP BY category;
```

### **Aggregate Functions**

Aggregate functions perform operations on groups of rows defined by the `**GROUP BY**` clause. They summarize the data within each group, providing valuable aggregated results.

**Common Aggregate Functions:**

- `**COUNT**`**:** Counts the number of rows in each group.
- `**SUM**`**:** Calculates the sum of values within each group.
- `**AVG**`**:** Computes the average value within each group.
- `**MIN**`**:** Finds the minimum value within each group.
- `**MAX**`**:** Identifies the maximum value within each group.

### **Filtering groups with** `**HAVING**`**:**

While the `**WHERE**` clause filters individual rows, the `**HAVING**` clause steps in after grouping to filter groups based on conditions applied to aggregate values. It's your tool for fine-tuning group-level criteria.

**Example:**

```SQL
-- Selecting departments with an average salary greater than $70,000
SELECT department, AVG(salary) AS average_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 70000;

```

### `**WHERE**` **meets** `**GROUP BY**`**:**

- Combining `**WHERE**` with `**GROUP BY**` allows you to filter rows before grouping, providing more control over the data included in each group.

**Example:**

```SQL
-- Selecting only sales above $100,000 for each product category
SELECT category, SUM(sales) AS total_sales
FROM products
WHERE sales > 100000
GROUP BY category;

```

### **Named Columns in** `**GROUP BY**`**:**

Similar to the `**GROUP BY**` and aggregate section, you can use column aliases in the `**GROUP BY**` clause for better readability.

**Example:**

```SQL
-- Using column aliases in GROUP BY for better readability
SELECT department AS dept, AVG(salary) AS average_salary
FROM employees
GROUP BY dept;

```

The harmonious interplay of `**WHERE**`, `**GROUP BY**`, and aggregate functions empowers you to sculpt your queries with precision, ensuring that the resulting data aligns seamlessly with your analytical goals.

## `**WHERE**` **and** `**HAVING**`

The `**WHERE**` clause is used to filter rows based on specific conditions, while the `**HAVING**` clause is used in conjunction with `**GROUP BY**` to filter groups based on conditions applied to aggregate values.

```SQL
SELECT column, AVG(another_column)
FROM table
WHERE condition
GROUP BY column
HAVING AVG(another_column) > threshold;
```

## **Subqueries**

Subqueries enable you to nest one query within another. They are useful for complex queries where you need the result of one query as input for another.

```SQL
SELECT column
FROM table
WHERE column IN (SELECT column FROM another_table WHERE condition);
```

Subqueries can be used in various parts of a SQL statement, such as the `**SELECT**`, `**FROM**`, and `**WHERE**` clauses.

Subqueries may also be named. This is especially useful in joins

```SQL
SELECT main_table.column1, main_table.column2, subquery.total_count
FROM main_table
JOIN (
  SELECT related_column, COUNT(*) AS total_count
  FROM related_table
  GROUP BY related_column
) AS subquery ON main_table.column1 = subquery.related_column;
```

## `**WITH**` **Common Table Expressions**

The `**WITH**` clause, also known as Common Table Expressions (CTE), allows you to define temporary result sets that can be referenced within the context of a larger query. It enhances the readability and reusability of complex queries.

```SQL
WITH temp_table AS (
  SELECT column
  FROM another_table
  WHERE condition
)

SELECT *
FROM main_table
JOIN temp_table ON main_table.column = temp_table.column;

```

The `**WITH**` clause simplifies queries by breaking them into more manageable parts.

## Going deeper…

- **String manipulation:**  
    String functions can be powerful for data cleansing or extraction. Example:  
    
    ```SQL
    SELECT CONCAT(first_name, ' ', last_name) AS full_name
    FROM employees;
    ```
    
    The `**CONCAT**` function combines the first and last names into a single column.
    
- `**CAST**`**:**  
    The  
    `**CAST**` function is handy for converting data types. Example:
    
    ```SQL
    SELECT CAST(numeric_column AS VARCHAR) AS string_column
    FROM table;
    ```
    
    Here, we're casting a numeric column as a VARCHAR.
    
- **Window functions:**
    
    Window functions operate across a set of table rows related to the current row. They provide a powerful way to perform calculations over a specified range of rows related to the current row. Window functions are typically used in conjunction with the `**OVER**` clause, which defines the window or set of rows the function operates on.
    
    ```SQL
    -- Example of calculating the running total of sales using a window function
    SELECT date, sales, SUM(sales) OVER (ORDER BY date) AS running_total
    FROM sales_data;
    ```
    
    Common window functions include `**ROW_NUMBER**`, `**RANK**`, `**DENSE_RANK**`, `**LAG**`, and `**LEAD**`. These functions offer advanced analytical capabilities, allowing you to derive insights from your data that go beyond basic aggregations.
    

# 🐍 Using SQL with Python

Python provides several powerful libraries for seamlessly integrating SQL into your data analysis workflow. Here, we'll explore two popular combinations: Pandas with Pyarrow and Pandas with DuckDB.

When to use each:

- Pandas + pyarrow are great default options (parquet, csv, excel)
- DuckDB can sometimes import databases that other options have trouble with and can work on larger-than-memory datasets
- SQLite is _lightest_ and comes installed out-of-the-box (but doesn’t have bells and whistles)
- SQLalchemy when you need to connect to a **_live database ⚡️_**

## **Pandas + Pyarrow**

Pandas is a widely-used data manipulation library in Python, and Pyarrow serves as a bridge between Pandas and Arrow, a cross-language development platform for in-memory data. This combination allows for efficient conversion and manipulation of large datasets.

### **Installation**

Ensure you have both Pandas and Pyarrow installed:

```Shell
pip install pandas
```

### **Using SQL with Pandas**

Pandas provides a convenient `**read_sql**` function that allows you to execute SQL queries and retrieve the results directly into a DataFrame.

```Python
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

# Example SQL query
sql_query = "SELECT column1, column2 FROM table WHERE condition;"

# Reading SQL query into Pandas DataFrame
df = pd.read_sql(sql_query, connection)
```

## **Pandas + DuckDB**

[DuckDB](https://duckdb.org/docs/guides/python/jupyter.html) is an in-memory analytical database that seamlessly integrates with Pandas. It is designed for analytical queries, making it a powerful companion for Pandas in data exploration and analysis.

### **Installation**

Install DuckDB using:

```Shell
pip install duckdb
```

### **Using SQL with DuckDB**

DuckDB provides a Pandas-friendly interface, making it easy to perform SQL queries directly on Pandas DataFrames.

```Python
import pandas as pd
import duckdb

# Create a dataframe from a file
my_df = pd.read_csv('example.csv')

# Reading SQL query into Pandas DataFrame
con = duckdb.connect()

# Create the table "my_table" from the DataFrame "my_df"
# Note: duckdb.sql connects to the default in-memory database connection
duckdb.sql("CREATE TABLE my_table AS SELECT * FROM my_df")

# Query from the database
results = duckdb.sql("SELECT * FROM my_table")
```

## SQLite

SQLite is a lightweight, file-based database engine that is often used for local development and small-scale applications. Python's standard library includes an SQLite module, making it easy to work with SQLite databases. It is ubiquitous, but also not a fully featured as pandas + pyarrow/duckdb.

### **Using SQLite with Python**

```Python
import sqlite3

# Connecting to an SQLite database (creates a new file if not exists)
conn = sqlite3.connect('example.db')

# Example SQL query
sql_query = "SELECT column1, column2 FROM table WHERE condition;"

# Reading SQL query into Pandas DataFrame
df = pd.read_sql_query(sql_query, conn)
```

## **SQLalchemy & external databases**

When working with larger databases, you may need to connect to external databases. Libraries like [SQLAlchemy](https://www.sqlalchemy.org/) provide a flexible and efficient way to interact with a variety of databases.

```Python
from sqlalchemy import create_engine

# Example connection to a PostgreSQL database
engine = create_engine('postgresql://user:password@localhost:5432/database')
df = pd.read_sql_query(sql_query, engine)
```

## 🐇 SQL Magic 🎩

The `ipython-sql` [package](https://pypi.org/project/ipython-sql/) (reimplemented by `jupysql`) provides a shortcut (%-commands are called “magic”) for querying SQLalchemy within notebooks:

- `%sql` inline queries
- `%%sql` for multi-line queries

```Python
# Install required libraries
%pip install pandas duckdb-engine ipython-sql

# Import necessary libraries
import pandas as pd
import duckdb
%load_ext sql

# Create a sample Pandas DataFrame
data = {'ID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 22, 35]}

df = pd.DataFrame(data)

# Connect to DuckDB and load the DataFrame into DuckDB
con = duckdb.connect(database=':memory:', read_only=False)
con.register('sample_data', df)

# Use SQL magic command to query the DuckDB database
%sql duckdb://localhost

# Show the content of the DuckDB database
%sql SELECT * FROM df
```

# 👩‍🏭 Emily’s tales from the trenches

> [!important]  
> Remember to update these to Emily’s latest  

![[Fake_Claims_Data_ET.csv]]

![[Fake_Cohort_ET.csv]]

![[SQLDraft_Tang011624.pptx]]

  

[[SQL Lecture Follow-up Notes]]

# Comparison to Pandas

![[sql_to_pandas.png]]

# 🦾 Exercise

1. (Recommended) SQL tutorial on the web - [https://sqlbolt.com](https://sqlbolt.com/)
2. (Optional) Practice using F1 data - [https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
3. 🌟 Coding exercise: loading the Chinook dataset and answering questions based on it

![[chinook_schema.png]]