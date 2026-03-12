---
lecture_number: 05
pdf: true
---

# Case Study: Decline in `product_event`

> There is a decline in the number of daily users engaging with <VALUABLE_FEATURE> for the mobile app. How would you go about investigating this issue?

> _Alternative prompt_
> A product leader approaches you with the belief that feature X is negatively affecting the product overall despite a lot of active usage.

## Investigation

Systematically define the problem through implicated systems, features, attributes, populations to identify potential causes. Eliminate possible causes through corroborating evidence.

1. Is this a “real” problem? Is it big enough to warrant an investigation? Can it be explained through known changes to the product, instrumentation, or funnel?
    1. Once the issue has been triaged and validated as “real”, document the investigation through an incident report, including a summary of current understanding at the top
    2. Loop in stakeholders with clear roles: Driver, Approver, Contributors, and Informed
    3. Set clear communication expectations for where/when updates will occur
2. Define the time period during which the decline occurred. Understand whether the decline occurs as step change(s) or gradual over time. Could it be normal variation? Was it a steady decline that was only just noticed?
3. Analyze the user segmentation to see if any specific groups were affected more than others. Where is the issue occurring in the user lifecycle? Which groups are involved? Devices? Locales?
4. Check the measurement process and ensure that tracking is functioning correctly (event emitted in browser, continues to appear on backend, ETL stats in between).
5. Do other, related metrics reflect the same change in expected upstream and downstream behavior? Are there noticeable fluctuations there that match?
6. Review recent changes to production and feature flags to identify recent updates or changes that might have impacted user engagement. If applicable, review marketing spend and tactics changes. Any recent issues filed related to the implicated area?
7. Context-specific: which unmeasured changes could impact the metric? Are there proxy measures that are either co-occurring or correlated with the decline or the unmeasured changes?
8. Identify steps to short-term remediation. If the issue is in tracking, define steps needed to fix and unit tests to monitor. If the issue is “real”, recommend changes or rollbacks to address the unintended impact.
9. Identify longer-term learnings. Flag any roadmapped changes that could have similar impact and list possible experiments that could validate product changes that could move the needle in the opposite direction.

# SQL Hands-on Screener

- Skills to be assessed:
    - Select/from/where
    - CTEs
    - Join
    - Group by
    - Filtering (where/having/in/not in)
    - Data types
    - Window functions

## Tables Available

- `activity`: User ID, event, context (view, org_id)

| user_id       | event_ts            | event               | view        | org_id |
| ------------- | ------------------- | ------------------- | ----------- | ------ |
| 0a1324bcf9901 | 2022-01-04 09:01:53 | context_noun_action | url_or_view |        |

- `membership`: User ID, org ID, date user added

|org_id|user_id|added_ts|
|---|---|---|
|0a1324bcaf0189|bfce917ae|2022-01-04 09:01:53 UTC|

- `organizations`: Org ID, org attributes (class = {free, smb, enterprise, edu, non-profit})

|org_id|class|
|---|---|
|0a1434bfae|'free’|
|bace14af|'enterprise’|

```SQL
WITH user_date AS (
  SELECT user_id
	   , event_ts::DATE AS active_dt
	FROM activity
   WHERE EXTRACT(WEEK FROM event_TS) = EXTRACT(WEEK FROM CURRENT_DATE() - INTERVAL '1 week')
   GROUP BY user_id, active_dt
), user_days AS (
  SELECT user_id
	   , COUNT(active_dt) AS active_days
	FROM user_date
)

SELECT COUNT(*) AS active_users
  FROM user_days
 WHERE active_days > 1
```


## 1a. Top 10 growing orgs (join, aggregate, filter)

> Find the top 10 growing organizations last month by number of added users

```SQL
SELECT COUNT(DISTINCT(user_id)) num_added
		 , org_id
  FROM membership
 WHERE EXTRACT('month' FROM added_ts) = EXTRACT('month' FROM (CURRENT_DATE() - '1 month'::INTERVAL))
 GROUP BY org_id
 ORDER BY num_added DESC
 LIMIT 10
```

### 1b. Top 9 Growing Orgs + “Other” (CTE, Self-join or Window or HAVING)

> Find the top 9 growing organizations last month by number of added users, plus a 10th row for “other”

```SQL
WITH added_users AS (
  SELECT num_added
       , org_id
       , RANK() OVER (ORDER BY num_added DESC) AS ranking
    FROM ( SELECT COUNT(DISTINCT(user_id)) AS num_added
                , org_id
             FROM membership
            WHERE EXTRACT(MONTH FROM added_ts) = EXTRACT(MONTH FROM (CURRENT_DATE() - INTERVAL '1 month'))
            GROUP BY org_id
         ) counts
)

SELECT num_added, org_id, ranking
  FROM ( SELECT num_added, org_id, ranking
           FROM added_users
          WHERE ranking < 10
          UNION ALL
         SELECT SUM(num_added) AS num_added
              , 'Other' AS org_id
              , 10 AS ranking
           FROM added_users
          WHERE ranking > 9
       ) AS combined
 ORDER BY ranking ASC
```

## 2. Active Users (grouping, CTE, aggregation)

> A user is “active” during a week if they have activity events on 2+ days during that week. How many active users were there last week?

## 3. Enterprise Champions (window Functions, Range joins)

> Identify enterprise champions: active users (as above) in more than one enterprise org, within one week of first being added as a user

```SQL
WITH enterprise_users AS (
  SELECT m.user_id
       , o.org_id
    FROM membership m
    JOIN organization o
   USING (org_id)
   WHERE o.class = 'enterprise'
), userorg_active_dt AS (
  SELECT user_id
       , org_id
       , event_ts::DATE AS active_dt
    FROM activity
   GROUP by user_id, org_id, active_dt
), userorg_min_dt AS (
  SELECT user_id
       , org_id
       , MIN(added_ts)::DATE min_dt
    FROM membership
)

SELECT uad.user_id
     , uad.org_id
  FROM userorg_active_dt uad
  JOIN userorg_min_dt
 USING (user_id, org_id)
 WHERE (uad.active_dt >= umd.min_dt) 
   AND (uad.active_dt <= (umd.min_dt + INTERVAL '7 days'))
 GROUP BY uad.user_id, uad.org_id
HAVING COUNT(DISTINCT(uad.active_dt)) > 1
```

## 4. Weekly Active Users (CTE, Lag, case)

> Users may be:  
> -   `new` (active this week, not active or snoozed last week)  
> -   `active` (active this week and last week)  
> -   `snoozed` (active last week, not this week)  
> -   `reactivated` (active this week, snoozed last week)  
> 
> Using the 2+ days of events to define ‘active’ during a week, how many users are in each category for the last 52 weeks?  

```SQL
WITH user_date AS (
  SELECT user_id
       , event_ts::DATE AS active_dt
    FROM activity
   WHERE DATE_TRUNC('week', event_TS) > DATE_TRUNC('week', CURRENT_DATE() - INTERVAL '52 weeks')
   GROUP BY user_id, active_dt
), user_week_base AS (
  SELECT user_id
       , DATE_TRUNC('week', active_dt) AS week
       , 'active' AS category
    FROM user_date
   GROUP BY user_id, week
   ORDER BY user_id, week
   HAVING COUNT(DISTINCT(active_dt)) >= 2
), uwb_snoozed AS (
  SELECT user_id, week, category
    FROM user_week_base
   UNION ALL ( SELECT user_id
                    , week + INTERVAL '1 week' AS week
                    , 'snoozed' AS category
               EXCEPT (SELECT user_id, week, 'snoozed' AS category
                         FROM user_week_base)
             )
  GROUP BY user_id, week, category
  ORDER BY user_id, week
), user_week_category AS (
  SELECT user_id, week
       , CASE WHEN (category = 'snoozed') THEN 'snoozed'
              WHEN (same_user AND lag_1wk) THEN 'active'
              WHEN (same_user AND NOT (lag_1wk OR lag_zzz)) THEN 'active'
              WHEN (NOT same_user) THEN 'new'
              WHEN (same_user AND NOT lag_zzz) THEN 'new'
              WHEN (lag_zzz) THEN 'reactivated'
              ELSE 'uncaught'
          END AS category
    FROM (SELECT user_id, week
               , (LAG(user_id, 1) OVER (ORDER BY user_id, week) = user_id) AS same_user
               , (LAG(week, 1) OVER (ORDER BY user_id, week) = week - INTERVAL '1 week') AS lag_1wk
               , (LAG(category, 1) OVER (ORDER BY user_id, week) = 'snoozed') AS lag_zzz 
            FROM uwb_snoozed) lag1
)

SELECT week
     , SUM(CASE WHEN category = 'new' THEN 1 ELSE 0 END) AS new_users
     , SUM(CASE WHEN category = 'active' THEN 1 ELSE 0 END) AS active_users
     , SUM(CASE WHEN category = 'snoozed' THEN 1 ELSE 0 END) AS snoozed_users
     , SUM(CASE WHEN category = 'reactivated' THEN 1 ELSE 0 END) AS reactivated_users
     , SUM(CASE WHEN category = 'uncaught' THEN 1 ELSE 0 END) AS uncaught_users
FROM user_weeks
GROUP BY week
ORDER BY week;
```

## 5. Joint Interview times

> Write a query for how much time two interviewers have spent doing joint interviews (together in the room) given…

1. Interviewers tag in and out of the interview room
2. Interviewers overlap for joint interviews, but not tag in at the same time
3. Assume perfect data quality and never more than two interviewers are present in the room (it’s a small room)

Table `interview`:

|Interviewer_ID|Start_time|Stop_time|
|---|---|---|
|`string`|`datetime`|`datetime`|

> [!important]  
> NOTE: Unsorted self-joins are slower `O(N^2)` than window functions `O(N log(N))`  

### 5a. Total Joint Interview time (self-join)

```SQL
SELECT
    SUM(
        LEAST(a.Stop_time, b.Stop_time) - GREATEST(a.Start_time, b.Start_time)
    ) AS Total_Joint_Interview_Time
FROM
    interview a
JOIN
    interview b
    ON a.Interviewer_ID != b.Interviewer_ID
    AND a.Start_time <= b.Stop_time
    AND b.Start_time <= a.Stop_time;
```

### 5b. Time per Pair of Interviewers

```SQL
SELECT
    a.Interviewer_ID AS Interviewer1,
    b.Interviewer_ID AS Interviewer2,
    SUM(
        LEAST(a.Stop_time, b.Stop_time) - GREATEST(a.Start_time, b.Start_time)
    ) AS Joint_Interview_Time
FROM
    interview a
JOIN
    interview b
    ON a.Interviewer_ID != b.Interviewer_ID
    AND a.Start_time <= b.Stop_time
    AND b.Start_time <= a.Stop_time
GROUP BY
    a.Interviewer_ID,
    b.Interviewer_ID;
```

### 5c. Total Time in Joint Interviews (window function)

```SQL
WITH tags AS (
	SELECT interviewer_id, start_time as tag_time, 1 as delta
	FROM interview
		UNION ALL
	SELECT interviewer_id, stop_time as tag_time, -1 as delta
	from interview
),
sorted_tags AS (
    SELECT 
        tag_time,
        SUM(delta) OVER (ORDER BY tag_time, delta DESC) as current_interviewers
    FROM tags
),
intervals AS (
    SELECT 
        tag_time as start_time,
        LEAD(tag_time) OVER (ORDER BY tag_time) as end_time,
        current_interviewers
    FROM sorted_tags
)

SELECT
    SUM(end_time - start_time) AS Total_Joint_Interview_Time
FROM intervals
WHERE current_interviewers = 2;
```

# Not yet Written up…

## TJ Murphy

I have done 100s of analytical SQL coding interviews. My standards are higher than most. Here’s what you need to know to pass –

» HOW IT WORKS
My interview is bring-your-own-database. I will describe some tables and ask you to write queries.

I will act as the query interface. You can ask me to look up function names at any time.

Just pick a database. I’m not impressed by breadth just tell me what you know.

» QUESTION 1: easy

You have 2 tables. The first has the names & ID of sports teams, the second is a list of sports matches with the winner & loser IDs.

Tell me the top 10 teams by win count for the 2024 season.

Required skills: select, group, filter, join

» QUESTION 1.5: medium

Same data as Q1 plus a new table that has win count by team for 2024.

Tell me the top 9 teams by wins for 2024, plus a 10th row for “all others”.

This question has 5+ valid solutions. I don’t care which you find.

Required skills: CTE/subquery, union, antijoin, self-join

» QUESTION 2: medium

You have 1 table, it represents customer purchases.

A repeat customer is someone who makes 2+ purchases, but 2 or more on the same day doesn’t count. Tell me how many repeat customers we have.

I want to see you break this into steps.

Required skills: CTE, grouping, aggregation

» QUESTION 3: hard

You have 2 tables, one is customer purchases, the second is marketing touches.

Tell me the count of customers with only organic orders on a last touch attribution basis.

Required skills: window functions, range joins

» EVALUATION
If you nail those three - you’ve passed. Super solid responses on 1 & 2 is all I need though.

If you’re struggling at one level, I’ll rotate in other questions at that difficulty. Otherwise if you breeze through it, I’ll keep ramping up the difficulty.

You should probably know:

• Select/from/where
• CTEs
• Join
• Group by
• Filtering techniques (where/having/in/not in)
• Basic data types

I don’t care about:

• DML
• Indexes
• temp tables
• Performance

» MOST IMPORTANT 
You need to learn how to do a technical interview. It’s not a quiz, it’s an audition. Silent and technically correct is a failure.

## Applied Statistics

(hat tip @[PhDemetri](https://twitter.com/PhDemetri/status/1673917856055001090)) Haven’t written this up yet

> Bill James is credited with creating sabermetrics (baseball analytics). In one of his early "Baseball Abstracts", Bill writes...  
> 
> 
> _If you see 15 games a year, there is a 40% chance that a .275 hitter will have more hits than a .300 hitter.  
> 
> 
> _Bill refers to players by their batting average (i.e. .275 means the hitter will hit the ball 275 times for every 1000 times they come at bat). The actual probability is quite smaller than that. Bill wrote this in the late 1970s without the ubiquity of computers to perform the simulations we can. It is quite plausible that Bill used a Normal approximation to arrive at this conclusion.  
> 
> Assuming that every batter appears 3 times per game for 15 games (for a total of 45 at bats), use a Normal approximation to estimate the probability that a 275 batter hits more hits than a .300 batter. Assume the batters are independent. You can use python to evaluate any complicated functions, but do not estimate the probability via simulation.
