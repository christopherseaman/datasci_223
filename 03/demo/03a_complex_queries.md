# Demo 3: Progressive CTEs and Flow Analytics

This demo builds SQL complexity progressively, from simple CTEs to multi-step flow analysis with visualizations. Run demo 1 first to set up the database.

## Step 0: Setup

```python
import altair as alt
import pandas as pd
import plotly.graph_objects as go

%load_ext sql
%sql duckdb:///demo_chinook.duckdb?access_mode=read_only
```

## Part A: Simple CTEs

### Step 1: Single CTE for customer lifetime value

A CTE (Common Table Expression) lets you name a subquery and reuse it. Start simple:

```python
%%sql customer_value <<
WITH customer_spend AS (
    SELECT CustomerId, 
           SUM(Total) AS lifetime_value,
           COUNT(*) AS order_count
    FROM invoices
    GROUP BY CustomerId
)
SELECT c.FirstName || ' ' || c.LastName AS customer_name,
       c.Country,
       ROUND(cs.lifetime_value, 2) AS lifetime_value,
       cs.order_count
FROM customers c
JOIN customer_spend cs ON c.CustomerId = cs.CustomerId
ORDER BY lifetime_value DESC
LIMIT 15;
```

### Ranked bar chart: Top customers by lifetime value

```python
df_customers = customer_value.DataFrame()

chart = (
    alt.Chart(df_customers)
    .mark_bar()
    .encode(
        x=alt.X('lifetime_value:Q', title='Lifetime Value ($)'),
        y=alt.Y('customer_name:N', sort='-x', title=None),
        color=alt.Color('Country:N', legend=alt.Legend(title='Country', columns=1)),
        tooltip=['customer_name', 'Country', 'lifetime_value', 'order_count']
    )
    .properties(title='Top 15 Customers by Lifetime Value', width=500, height=350)
)

chart
```

## Part B: Chained CTEs

### Step 2: Two CTEs for artist revenue ranking

Chain CTEs to build logic step-by-step:

```python
%%sql artist_revenue <<
WITH track_revenue AS (
    SELECT t.TrackId,
           t.Name AS track_name,
           al.AlbumId,
           SUM(ii.UnitPrice * ii.Quantity) AS track_total
    FROM invoice_items ii
    JOIN tracks t ON ii.TrackId = t.TrackId
    JOIN albums al ON t.AlbumId = al.AlbumId
    GROUP BY t.TrackId, t.Name, al.AlbumId
),
artist_totals AS (
    SELECT ar.ArtistId,
           ar.Name AS artist_name,
           COUNT(DISTINCT tr.TrackId) AS tracks_sold,
           ROUND(SUM(tr.track_total), 2) AS total_revenue
    FROM track_revenue tr
    JOIN albums al ON tr.AlbumId = al.AlbumId
    JOIN artists ar ON al.ArtistId = ar.ArtistId
    GROUP BY ar.ArtistId, ar.Name
)
SELECT artist_name, tracks_sold, total_revenue
FROM artist_totals
ORDER BY total_revenue DESC
LIMIT 20;
```

### Bubble chart: Artist revenue vs tracks sold

```python
df_artists = artist_revenue.DataFrame()

chart = (
    alt.Chart(df_artists)
    .mark_circle()
    .encode(
        x=alt.X('tracks_sold:Q', title='Unique Tracks Sold'),
        y=alt.Y('total_revenue:Q', title='Total Revenue ($)'),
        size=alt.Size('total_revenue:Q', legend=None, scale=alt.Scale(range=[50, 500])),
        color=alt.value('#e45756'),
        tooltip=['artist_name', 'tracks_sold', 'total_revenue']
    )
    .properties(title='Artist Performance: Tracks Sold vs Revenue', width=500, height=350)
)

# Add artist labels for top performers
text = (
    alt.Chart(df_artists.head(5))
    .mark_text(align='left', dx=10, fontSize=11)
    .encode(
        x='tracks_sold:Q',
        y='total_revenue:Q',
        text='artist_name:N'
    )
)

chart + text
```

## Part C: CTEs with Window Functions

### Step 3: Running totals and month-over-month growth

Combine CTEs with window functions for time-series analysis:

```python
%%sql monthly_growth <<
WITH monthly_revenue AS (
    SELECT DATE_TRUNC('month', InvoiceDate) AS month,
           SUM(Total) AS revenue
    FROM invoices
    GROUP BY month
),
with_growth AS (
    SELECT month,
           revenue,
           LAG(revenue) OVER (ORDER BY month) AS prev_month,
           SUM(revenue) OVER (ORDER BY month) AS cumulative_revenue
    FROM monthly_revenue
)
SELECT month,
       ROUND(revenue, 2) AS revenue,
       ROUND(cumulative_revenue, 2) AS cumulative_revenue,
       ROUND(100.0 * (revenue - prev_month) / NULLIF(prev_month, 0), 1) AS pct_growth
FROM with_growth
ORDER BY month;
```

### Growth visualization: Monthly revenue with percent change

```python
df_growth = monthly_growth.DataFrame()

# Bar chart colored by growth rate
bars = alt.Chart(df_growth).mark_bar().encode(
    x=alt.X('month:T', title=None),
    y=alt.Y('revenue:Q', title='Monthly Revenue ($)'),
    color=alt.Color('pct_growth:Q', 
        scale=alt.Scale(scheme='redyellowgreen', domain=[-50, 50]),
        legend=alt.Legend(title='% Growth')),
    tooltip=['month:T', 'revenue', 'pct_growth', 'cumulative_revenue']
)

# Cumulative line overlay
cumulative_line = alt.Chart(df_growth).mark_line(
    color='#333', strokeWidth=2, strokeDash=[4, 2]
).encode(
    x='month:T',
    y=alt.Y('cumulative_revenue:Q', title='Cumulative Revenue')
)

# Combine with dual axis
alt.layer(bars, cumulative_line).resolve_scale(
    y='independent'
).properties(
    title='Monthly Revenue (bars) with Cumulative Total (dashed line)',
    width=650, height=300
)
```

### Step 4: Rank artists within each genre (PARTITION BY)

```python
%%sql top_per_genre <<
WITH artist_genre_revenue AS (
    SELECT g.Name AS genre,
           ar.Name AS artist_name,
           ROUND(SUM(ii.UnitPrice * ii.Quantity), 2) AS revenue
    FROM invoice_items ii
    JOIN tracks t ON ii.TrackId = t.TrackId
    JOIN genres g ON t.GenreId = g.GenreId
    JOIN albums al ON t.AlbumId = al.AlbumId
    JOIN artists ar ON al.ArtistId = ar.ArtistId
    GROUP BY g.Name, ar.Name
),
ranked AS (
    SELECT genre, artist_name, revenue,
           ROW_NUMBER() OVER (PARTITION BY genre ORDER BY revenue DESC) AS rank
    FROM artist_genre_revenue
)
SELECT genre, artist_name, revenue, rank
FROM ranked
WHERE rank <= 3
ORDER BY genre, rank;
```

### Faceted bar chart: Top 3 artists per genre

```python
df_per_genre = top_per_genre.DataFrame()

top_genres = ['Rock', 'Metal', 'Alternative & Punk', 'Latin', 'Jazz', 'Blues']
filtered = df_per_genre[df_per_genre['genre'].isin(top_genres)]

chart = (
    alt.Chart(filtered)
    .mark_bar()
    .encode(
        x=alt.X('revenue:Q', title='Revenue ($)'),
        y=alt.Y('artist_name:N', sort='-x', title=None),
        color=alt.Color('rank:O', 
            scale=alt.Scale(domain=[1, 2, 3], range=['#f58518', '#54a24b', '#4c78a8']),
            legend=alt.Legend(title='Rank')),
        tooltip=['genre', 'artist_name', 'revenue', 'rank']
    )
    .properties(width=150, height=100)
    .facet(facet='genre:N', columns=3)
    .properties(title='Top 3 Artists by Revenue in Each Genre')
)

chart
```

## Part D: Complex Multi-CTE Flow Analysis

### Step 5: Revenue flow from Genre → Country

This query traces how revenue flows from music genres to customer countries:

```python
%%sql genre_country_flow <<
WITH purchases AS (
    SELECT g.Name AS genre,
           c.Country AS country,
           ii.UnitPrice * ii.Quantity AS amount
    FROM invoice_items ii
    JOIN tracks t ON ii.TrackId = t.TrackId
    JOIN genres g ON t.GenreId = g.GenreId
    JOIN invoices i ON ii.InvoiceId = i.InvoiceId
    JOIN customers c ON i.CustomerId = c.CustomerId
),
flow_summary AS (
    SELECT genre, country,
           ROUND(SUM(amount), 2) AS flow_value,
           COUNT(*) AS purchase_count
    FROM purchases
    GROUP BY genre, country
)
SELECT genre, country, flow_value, purchase_count
FROM flow_summary
WHERE flow_value >= 5
ORDER BY flow_value DESC;
```

### Heatmap: Genre to Country flow

```python
df_flow = genre_country_flow.DataFrame()

top_genres = df_flow.groupby('genre')['flow_value'].sum().nlargest(6).index.tolist()
top_countries = df_flow.groupby('country')['flow_value'].sum().nlargest(8).index.tolist()

filtered_flow = df_flow[
    (df_flow['genre'].isin(top_genres)) & 
    (df_flow['country'].isin(top_countries))
]

chart = (
    alt.Chart(filtered_flow)
    .mark_rect()
    .encode(
        x=alt.X('country:N', title='Customer Country', sort=top_countries),
        y=alt.Y('genre:N', title='Music Genre', sort=top_genres),
        color=alt.Color('flow_value:Q', 
            scale=alt.Scale(scheme='orangered'),
            legend=alt.Legend(title='Revenue ($)')),
        tooltip=['genre', 'country', 'flow_value', 'purchase_count']
    )
    .properties(title='Revenue Flow: Genre → Country', width=450, height=300)
)

chart
```

### Step 6: Full customer journey funnel with 4 CTEs

Track the complete path from artist to final sale:

```python
%%sql customer_journey <<
WITH artist_albums AS (
    SELECT ar.Name AS artist_name,
           COUNT(DISTINCT al.AlbumId) AS album_count
    FROM artists ar
    LEFT JOIN albums al ON ar.ArtistId = al.ArtistId
    GROUP BY ar.Name
),
album_tracks AS (
    SELECT ar.Name AS artist_name,
           COUNT(DISTINCT t.TrackId) AS track_count
    FROM artists ar
    JOIN albums al ON ar.ArtistId = al.ArtistId
    JOIN tracks t ON al.AlbumId = t.AlbumId
    GROUP BY ar.Name
),
track_sales AS (
    SELECT ar.Name AS artist_name,
           COUNT(DISTINCT ii.InvoiceLineId) AS sales_count,
           ROUND(SUM(ii.UnitPrice * ii.Quantity), 2) AS revenue
    FROM artists ar
    JOIN albums al ON ar.ArtistId = al.ArtistId
    JOIN tracks t ON al.AlbumId = t.AlbumId
    JOIN invoice_items ii ON t.TrackId = ii.TrackId
    GROUP BY ar.Name
),
funnel AS (
    SELECT aa.artist_name,
           aa.album_count,
           COALESCE(atr.track_count, 0) AS track_count,
           COALESCE(ts.sales_count, 0) AS sales_count,
           COALESCE(ts.revenue, 0) AS revenue
    FROM artist_albums aa
    LEFT JOIN album_tracks atr ON aa.artist_name = atr.artist_name
    LEFT JOIN track_sales ts ON aa.artist_name = ts.artist_name
)
SELECT * FROM funnel
WHERE sales_count > 0
ORDER BY revenue DESC
LIMIT 15;
```

### Funnel comparison: Albums → Tracks → Sales

```python
df_journey = customer_journey.DataFrame()

funnel_data = []
for _, row in df_journey.iterrows():
    funnel_data.append({'artist': row['artist_name'], 'stage': '1. Albums', 'count': row['album_count']})
    funnel_data.append({'artist': row['artist_name'], 'stage': '2. Tracks', 'count': row['track_count']})
    funnel_data.append({'artist': row['artist_name'], 'stage': '3. Sales', 'count': row['sales_count']})

funnel_df = pd.DataFrame(funnel_data)

top_5 = df_journey['artist_name'].head(5).tolist()
funnel_df = funnel_df[funnel_df['artist'].isin(top_5)]

chart = (
    alt.Chart(funnel_df)
    .mark_bar()
    .encode(
        x=alt.X('stage:N', title=None, sort=['1. Albums', '2. Tracks', '3. Sales']),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('artist:N', legend=alt.Legend(title='Artist')),
        xOffset='artist:N',
        tooltip=['artist', 'stage', 'count']
    )
    .properties(title='Artist Catalog Funnel: Albums → Tracks → Sales', width=450, height=300)
)

chart
```

### Step 7: Sankey diagram (Genre → Artist → Country)

Build a complete flow visualization showing how revenue flows through the catalog:

```python
%%sql full_flow <<
WITH detailed_sales AS (
    SELECT g.Name AS genre,
           ar.Name AS artist_name,
           c.Country AS country,
           ii.UnitPrice * ii.Quantity AS amount
    FROM invoice_items ii
    JOIN tracks t ON ii.TrackId = t.TrackId
    JOIN genres g ON t.GenreId = g.GenreId
    JOIN albums al ON t.AlbumId = al.AlbumId
    JOIN artists ar ON al.ArtistId = ar.ArtistId
    JOIN invoices i ON ii.InvoiceId = i.InvoiceId
    JOIN customers c ON i.CustomerId = c.CustomerId
),
genre_artist AS (
    SELECT genre AS source, artist_name AS target, 
           'genre_to_artist' AS link_type,
           ROUND(SUM(amount), 2) AS value
    FROM detailed_sales
    GROUP BY genre, artist_name
    HAVING SUM(amount) >= 15
),
artist_country AS (
    SELECT artist_name AS source, country AS target,
           'artist_to_country' AS link_type,
           ROUND(SUM(amount), 2) AS value
    FROM detailed_sales
    GROUP BY artist_name, country
    HAVING SUM(amount) >= 10
)
SELECT * FROM genre_artist
UNION ALL
SELECT * FROM artist_country
ORDER BY link_type, value DESC;
```

### Sankey: Revenue flow visualization

```python
df_full_flow = full_flow.DataFrame()

all_nodes = pd.concat([df_full_flow['source'], df_full_flow['target']]).unique().tolist()
node_indices = {node: i for i, node in enumerate(all_nodes)}

source_indices = [node_indices[s] for s in df_full_flow['source']]
target_indices = [node_indices[t] for t in df_full_flow['target']]
values = df_full_flow['value'].tolist()

genres = df_full_flow[df_full_flow['link_type'] == 'genre_to_artist']['source'].unique()
countries = df_full_flow[df_full_flow['link_type'] == 'artist_to_country']['target'].unique()
node_colors = ['#4c78a8' if n in genres else '#54a24b' if n in countries else '#f58518' for n in all_nodes]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=node_colors
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values,
        color='rgba(200, 200, 200, 0.5)'
    )
)])

fig.update_layout(
    title_text="Revenue Flow: Genre → Artist → Country",
    font_size=10,
    height=500
)

fig
```

## Step 8: Query performance with EXPLAIN

```python
%%sql
EXPLAIN
SELECT g.Name, ar.Name, SUM(ii.UnitPrice * ii.Quantity)
FROM invoice_items ii
JOIN tracks t ON ii.TrackId = t.TrackId
JOIN genres g ON t.GenreId = g.GenreId
JOIN albums al ON t.AlbumId = al.AlbumId
JOIN artists ar ON al.ArtistId = ar.ArtistId
GROUP BY g.Name, ar.Name;
```

## Recap: CTE Complexity Progression

| Level | Pattern | Example |
|-------|---------|---------|
| Simple | Single CTE | Customer lifetime value |
| Chained | Multiple sequential CTEs | Artist revenue via track → album → artist |
| With windows | CTE + window functions | Monthly growth with LAG |
| Complex | 3+ CTEs building a funnel | Full catalog journey analysis |

CTEs make complex queries readable by giving each step a name. Start simple and add layers as needed.
