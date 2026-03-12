# Demo 2: SQL Basics with Visual Analytics

This demo practices core SQL clauses on the Chinook database, visualizing results with varied chart types. Run demo 1 first to set up the database.

## Step 0: Setup

```python
import altair as alt
import pandas as pd

%load_ext sql
%sql duckdb:///demo_chinook.duckdb?access_mode=read_only
```

## Step 1: SELECT and explore tracks

```python
%%sql
SELECT TrackId, Name, Milliseconds, UnitPrice
FROM tracks
LIMIT 10;
```

### The longest tracks in the catalog

```python
%%sql longest_tracks <<
SELECT Name AS track_name,
       ROUND(Milliseconds / 60000.0, 1) AS duration_minutes
FROM tracks
ORDER BY Milliseconds DESC
LIMIT 15;
```

```python
df_longest = longest_tracks.DataFrame()

chart = (
    alt.Chart(df_longest)
    .mark_bar(color='#4c78a8')
    .encode(
        x=alt.X('duration_minutes:Q', title='Duration (minutes)'),
        y=alt.Y('track_name:N', sort='-x', title=None),
        tooltip=['track_name', 'duration_minutes']
    )
    .properties(title='15 Longest Tracks in the Catalog', width=450, height=350)
)

chart
```

## Step 2: WHERE + ORDER BY for filtering

```python
%%sql
SELECT Name, UnitPrice, ROUND(Milliseconds / 1000.0, 0) AS seconds
FROM tracks
WHERE UnitPrice >= 1.50
ORDER BY UnitPrice DESC, Milliseconds DESC
LIMIT 10;
```

## Step 3: DISTINCT and unique values

```python
%%sql
SELECT DISTINCT Country
FROM customers
ORDER BY Country;
```

## Step 4: GROUP BY with genre breakdown

```python
%%sql genre_counts <<
SELECT g.Name AS genre, COUNT(*) AS track_count
FROM tracks t
JOIN genres g ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY track_count DESC;
```

### Horizontal bar: Genre distribution

```python
df_genres = genre_counts.DataFrame()

alt.Chart(df_genres).mark_bar().encode(
    x=alt.X('track_count:Q', title='Number of Tracks'),
    y=alt.Y('genre:N', sort='-x', title=None),
    color=alt.Color('track_count:Q', scale=alt.Scale(scheme='blues'), legend=None),
    tooltip=['genre', 'track_count']
).properties(title='Track Distribution by Genre', width=450, height=400)
```

## Step 5: CASE WHEN for categorization

```python
%%sql track_lengths <<
SELECT
    CASE
        WHEN Milliseconds < 180000 THEN '< 3 min'
        WHEN Milliseconds < 300000 THEN '3-5 min'
        WHEN Milliseconds < 600000 THEN '5-10 min'
        ELSE '10+ min'
    END AS length_category,
    COUNT(*) AS track_count
FROM tracks
GROUP BY length_category;
```

### Bar chart with custom order

```python
df_lengths = track_lengths.DataFrame()
category_order = ['< 3 min', '3-5 min', '5-10 min', '10+ min']

chart = (
    alt.Chart(df_lengths)
    .mark_bar()
    .encode(
        x=alt.X('length_category:N', sort=category_order, title='Track Length'),
        y=alt.Y('track_count:Q', title='Number of Tracks'),
        color=alt.Color('length_category:N', 
            scale=alt.Scale(domain=category_order, range=['#9ecae1', '#6baed6', '#3182bd', '#08519c']),
            legend=None),
        tooltip=['length_category', 'track_count']
    )
    .properties(title='Tracks by Duration Category', width=350, height=250)
)

chart
```

## Step 6: Aggregations with HAVING

```python
%%sql country_genre_stats <<
SELECT c.Country AS country,
       g.Name AS genre,
       COUNT(*) AS purchases,
       ROUND(SUM(ii.UnitPrice * ii.Quantity), 2) AS revenue
FROM invoice_items ii
JOIN invoices i ON ii.InvoiceId = i.InvoiceId
JOIN customers c ON i.CustomerId = c.CustomerId
JOIN tracks t ON ii.TrackId = t.TrackId
JOIN genres g ON t.GenreId = g.GenreId
GROUP BY c.Country, g.Name
HAVING COUNT(*) >= 3
ORDER BY revenue DESC;
```

### Faceted bubbles: What genres sell where?

```python
df_cg = country_genre_stats.DataFrame()

# Top 6 countries and top 6 genres by total revenue
top_countries = df_cg.groupby('country')['revenue'].sum().nlargest(6).index.tolist()
top_genres = df_cg.groupby('genre')['revenue'].sum().nlargest(6).index.tolist()
filtered = df_cg[(df_cg['country'].isin(top_countries)) & (df_cg['genre'].isin(top_genres))]

alt.Chart(filtered).mark_circle().encode(
    x=alt.X('country:N', title=None, sort=top_countries),
    y=alt.Y('genre:N', title=None, sort=top_genres),
    size=alt.Size('revenue:Q', scale=alt.Scale(range=[20, 400]), legend=alt.Legend(title='Revenue')),
    color=alt.Color('purchases:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title='Purchases')),
    tooltip=['country', 'genre', 'purchases', 'revenue']
).properties(title='Genre Popularity by Country', width=400, height=300)
```

## Step 7: Time-based analysis with DATE_TRUNC

```python
%%sql monthly_sales <<
SELECT DATE_TRUNC('month', InvoiceDate) AS month,
       COUNT(*) AS invoice_count,
       ROUND(SUM(Total), 2) AS revenue
FROM invoices
GROUP BY month
ORDER BY month;
```

### Ridgeline-style: Revenue by year

```python
df_monthly = monthly_sales.DataFrame()
df_monthly['year'] = pd.to_datetime(df_monthly['month']).dt.year
df_monthly['month_num'] = pd.to_datetime(df_monthly['month']).dt.month

alt.Chart(df_monthly).transform_filter(
    alt.datum.year >= 2009
).mark_area(
    interpolate='monotone', fillOpacity=0.6, stroke='white', strokeWidth=0.5
).encode(
    x=alt.X('month_num:O', title='Month', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('revenue:Q', title='Revenue ($)', stack=None),
    color=alt.Color('year:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title='Year')),
    row=alt.Row('year:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left')),
    tooltip=['year', 'month_num', 'revenue', 'invoice_count']
).properties(width=500, height=60)
```

## Step 8: Heatmap for invoice patterns

```python
%%sql invoice_heatmap <<
SELECT 
    EXTRACT(YEAR FROM InvoiceDate) AS year,
    EXTRACT(MONTH FROM InvoiceDate) AS month,
    COUNT(*) AS invoice_count,
    ROUND(SUM(Total), 2) AS revenue
FROM invoices
GROUP BY year, month
ORDER BY year, month;
```

### Heatmap with values: Revenue by Year × Month

```python
df_heatmap = invoice_heatmap.DataFrame()

heatmap = alt.Chart(df_heatmap).mark_rect().encode(
    x=alt.X('month:O', title='Month'),
    y=alt.Y('year:O', title='Year', sort='descending'),
    color=alt.Color('revenue:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title='Revenue')),
    tooltip=['year', 'month', 'invoice_count', 'revenue']
)

text = alt.Chart(df_heatmap).mark_text(fontSize=9, color='white').encode(
    x='month:O',
    y=alt.Y('year:O', sort='descending'),
    text=alt.Text('revenue:Q', format='.0f')
)

(heatmap + text).properties(title='Revenue Heatmap: Year × Month', width=500, height=220)
```

## Step 9: COUNT(DISTINCT) for unique customers

```python
%%sql
SELECT COUNT(*) AS total_invoices,
       COUNT(DISTINCT CustomerId) AS unique_customers
FROM invoices;
```

```python
%%sql customer_frequency <<
SELECT CustomerId, COUNT(*) AS order_count
FROM invoices
GROUP BY CustomerId
ORDER BY order_count DESC;
```

### Lollipop chart: Top spenders by country

```python
%%sql top_spenders <<
SELECT c.FirstName || ' ' || c.LastName AS customer,
       c.Country AS country,
       COUNT(*) AS orders,
       ROUND(SUM(i.Total), 2) AS total_spent
FROM customers c
JOIN invoices i ON c.CustomerId = i.CustomerId
GROUP BY c.CustomerId, c.FirstName, c.LastName, c.Country
ORDER BY total_spent DESC
LIMIT 20;
```

```python
df_top = top_spenders.DataFrame()

# Lollipop: line from 0 to value + circle at end
lines = alt.Chart(df_top).mark_rule(strokeWidth=2).encode(
    x=alt.X('total_spent:Q', title='Total Spent ($)'),
    y=alt.Y('customer:N', sort='-x', title=None),
    color=alt.Color('country:N', legend=alt.Legend(title='Country'))
)

points = alt.Chart(df_top).mark_circle(size=100).encode(
    x='total_spent:Q',
    y=alt.Y('customer:N', sort='-x'),
    color='country:N',
    tooltip=['customer', 'country', 'orders', 'total_spent']
)

(lines + points).properties(title='Top 20 Customers by Spending', width=450, height=400)
```

## Step 10: LIKE/ILIKE for pattern matching

```python
%%sql
SELECT t.Name AS track_name, ar.Name AS artist_name, g.Name AS genre
FROM tracks t
JOIN albums al ON t.AlbumId = al.AlbumId
JOIN artists ar ON al.ArtistId = ar.ArtistId
JOIN genres g ON t.GenreId = g.GenreId
WHERE t.Name ILIKE '%love%'
ORDER BY ar.Name
LIMIT 15;
```
