# Demo 1: Setup Chinook and Explore the Data Landscape

This demo sets up DuckDB in a notebook, imports the Chinook SQLite database, and visualizes the data landscape. The Chinook database models a digital music store with artists, albums, tracks, customers, and sales.

## Schema overview

![Chinook schema](media/chinook_schema.png)

The schema flows from left to right:
- **Catalog**: genres, media_types, artists, albums, tracks
- **Sales**: invoices, invoice_items
- **People**: customers, employees

Optional: open the full PDF diagram for zooming: `media/chinook-database-diagram.pdf`.

## Step 1: Setup

Install packages from `lectures/03/demo/requirements.txt` in your terminal, then run:

```python
import altair as alt
import pandas as pd

%load_ext sql
%sql duckdb:///demo_chinook.duckdb
```

## Step 2: Attach the Chinook SQLite database

```python
%%sql
INSTALL sqlite_scanner;
LOAD sqlite_scanner;
ATTACH 'chinook.sqlite' AS chinook (TYPE SQLITE);
```

```python
%%sql
-- List tables from the attached database
SELECT table_name FROM information_schema.tables 
WHERE table_catalog = 'chinook';
```

## Step 3: Import core tables into DuckDB

```python
%%sql
-- Drop existing tables to avoid errors on re-running
DROP TABLE IF EXISTS genres;
DROP TABLE IF EXISTS media_types;
DROP TABLE IF EXISTS artists;
DROP TABLE IF EXISTS albums;
DROP TABLE IF EXISTS tracks;
DROP TABLE IF EXISTS playlists;
DROP TABLE IF EXISTS playlist_track;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS invoice_items;

-- Import all tables for richer analysis
CREATE TABLE genres AS SELECT * FROM chinook.Genres;
CREATE TABLE media_types AS SELECT * FROM chinook.Media_Types;
CREATE TABLE artists AS SELECT * FROM chinook.Artists;
CREATE TABLE albums AS SELECT * FROM chinook.Albums;
CREATE TABLE tracks AS SELECT * FROM chinook.Tracks;
CREATE TABLE playlists AS SELECT * FROM chinook.Playlists;
CREATE TABLE playlist_track AS SELECT * FROM chinook.Playlist_Track;
CREATE TABLE customers AS SELECT * FROM chinook.Customers;
CREATE TABLE employees AS SELECT * FROM chinook.Employees;
CREATE TABLE invoices AS SELECT * FROM chinook.Invoices;
CREATE TABLE invoice_items AS SELECT * FROM chinook.Invoice_Items;
```

```python
%%sql
SELECT 'genres' AS table_name, COUNT(*) AS rows FROM genres
UNION ALL SELECT 'media_types', COUNT(*) FROM media_types
UNION ALL SELECT 'artists', COUNT(*) FROM artists
UNION ALL SELECT 'albums', COUNT(*) FROM albums
UNION ALL SELECT 'tracks', COUNT(*) FROM tracks
UNION ALL SELECT 'playlists', COUNT(*) FROM playlists
UNION ALL SELECT 'playlist_track', COUNT(*) FROM playlist_track
UNION ALL SELECT 'customers', COUNT(*) FROM customers
UNION ALL SELECT 'employees', COUNT(*) FROM employees
UNION ALL SELECT 'invoices', COUNT(*) FROM invoices
UNION ALL SELECT 'invoice_items', COUNT(*) FROM invoice_items
ORDER BY rows DESC;
```

## Step 4: Visualize the data landscape

```python
%%sql table_counts <<
SELECT 'genres' AS table_name, COUNT(*) AS rows FROM genres
UNION ALL SELECT 'media_types', COUNT(*) FROM media_types
UNION ALL SELECT 'artists', COUNT(*) FROM artists
UNION ALL SELECT 'albums', COUNT(*) FROM albums
UNION ALL SELECT 'tracks', COUNT(*) FROM tracks
UNION ALL SELECT 'playlists', COUNT(*) FROM playlists
UNION ALL SELECT 'playlist_track', COUNT(*) FROM playlist_track
UNION ALL SELECT 'customers', COUNT(*) FROM customers
UNION ALL SELECT 'employees', COUNT(*) FROM employees
UNION ALL SELECT 'invoices', COUNT(*) FROM invoices
UNION ALL SELECT 'invoice_items', COUNT(*) FROM invoice_items
ORDER BY rows DESC;
```

```python
# Convert to DataFrame and ensure numeric type
df_counts = table_counts.DataFrame()
df_counts['rows'] = pd.to_numeric(df_counts['rows'])

# Categorize tables by domain
domain_map = {
    'tracks': 'Catalog', 'albums': 'Catalog', 'artists': 'Catalog',
    'genres': 'Catalog', 'media_types': 'Catalog', 'playlists': 'Catalog',
    'playlist_track': 'Catalog', 'customers': 'People', 'employees': 'People',
    'invoices': 'Sales', 'invoice_items': 'Sales'
}
df_counts['domain'] = df_counts['table_name'].map(domain_map)

df_counts
```

```python
alt.Chart(df_counts).mark_bar().encode(
    x=alt.X('rows:Q', title='Row count'),
    y=alt.Y('table_name:N', sort='-x', title=None),
    color=alt.Color('domain:N', 
        scale=alt.Scale(domain=['Catalog', 'Sales', 'People'],
                      range=['#4c78a8', '#f58518', '#54a24b']),
        legend=alt.Legend(title='Domain')),
    tooltip=['table_name', 'rows', 'domain']
).properties(title='Chinook Database: Table Sizes by Domain', width=500, height=300)
```

## Step 5: Quick peek at the catalog

Let's see what music we're working with:

```python
%%sql
-- Top 5 genres by track count
SELECT g.Name AS genre, COUNT(*) AS track_count
FROM tracks t
JOIN genres g ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY track_count DESC
LIMIT 5;
```

```python
%%sql
-- Top 5 artists by album count
SELECT ar.Name AS artist, COUNT(*) AS album_count
FROM albums al
JOIN artists ar ON al.ArtistId = ar.ArtistId
GROUP BY ar.Name
ORDER BY album_count DESC
LIMIT 5;
```

## Step 6: Mini CSV and Parquet round-trip

```python
%%sql
COPY (SELECT * FROM tracks LIMIT 50)
TO 'tracks_sample.csv' (HEADER, DELIMITER ',');

COPY (SELECT * FROM tracks LIMIT 50)
TO 'tracks_sample.parquet' (FORMAT PARQUET);
```

```python
%%sql
SELECT Name, Milliseconds, UnitPrice FROM read_csv_auto('tracks_sample.csv') LIMIT 3;
```

```python
%%sql
SELECT Name, Milliseconds, UnitPrice FROM read_parquet('tracks_sample.parquet') LIMIT 3;
```
