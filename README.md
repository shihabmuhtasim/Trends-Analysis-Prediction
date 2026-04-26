# Analyzing and Predicting Trends in Contemporary Social Discourse through Hashtag Campaigns

This repository contains the code, experiments, and outputs for the research project:

**Analyzing and Predicting Trends in Contemporary Social Discourse through Hashtag Campaigns**

The project combines **network science**, **graph-based community detection**, **natural language processing (NLP)**, **sentiment analysis**, and **time-series forecasting** to analyze and predict hashtag-driven social media trends. The main case study focuses on the propagation of **#Bitcoin** on Twitter/X.

---

## Overview

Hashtags play an important role in organizing online conversations and spreading information across social media platforms. This project studies how a hashtag campaign propagates through user interactions and how future activity related to the hashtag can be predicted.

The research pipeline includes:

- Tweet preprocessing and feature engineering
- Hashtag-level exploratory data analysis
- Retweet-based user interaction graph construction
- Community detection using Greedy Modularity Maximization
- Community leader identification
- Sentiment analysis of community leaders using ALBERT
- User-level hourly activity forecasting using LSTM models
- Fusion ensemble modeling using machine learning regressors

The main objective is to understand the structural and behavioral patterns behind hashtag propagation and to predict future user activity associated with a trending hashtag.

---

## Research Paper

**Title:**  
Analyzing and Predicting Trends in Contemporary Social Discourse through Hashtag Campaigns

**Authors:**  
Shihab Muhtasim, Raiyan Wasi Siddiky, Monirul Haque, and Farig Yousuf Sadeque

**Institution:**  
Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh

**Corresponding Author:**  
Shihab Muhtasim  
Email: shihabmuhtasim.cs@gmail.com

---

## Key Contributions

- Constructs a retweet-based user interaction network for hashtag propagation analysis.
- Detects user communities using Greedy Modularity Maximization.
- Identifies community leaders using graph connectivity patterns.
- Applies ALBERT-based sentiment analysis to study the sentiment of influential users.
- Builds two LSTM models for hourly activity prediction:
  - A multivariate numerical LSTM model
  - A tweet-content-based LSTM model using BERT vector representations
- Combines model outputs through fusion ensemble regressors.
- Evaluates prediction performance using MSE, RMSE, MAE, and R-squared.

---

## Dataset

The project uses the **Bitcoin Tweets 2022** dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/kumari2000/bitcoin-tweets-2022

The dataset contains tweets and retweets related to Bitcoin collected from **September 15–17, 2022**.

### Dataset Summary

| Property | Description |
|---|---|
| Platform | Twitter/X |
| Topic | Bitcoin-related tweets |
| Main hashtag | `#Bitcoin` |
| Original size | 337,702 tweets and retweets |
| Time span | September 15–17, 2022 |
| Original columns | 19 |
| Processed rows | 253,516 |
| Processed columns | 11 |

Important columns include:

- `Tweet Id`
- `Tweet Content`
- `Retweets Received`
- `Likes Received`
- `Username`
- `Followers`
- `Following`
- `Datetime`

---

## Methodology

The full workflow is divided into four major parts:

1. **Data preprocessing and analysis**
2. **Network construction and community analysis**
3. **Sentiment analysis of community leaders**
4. **User activity prediction**

---

## 1. Data Preprocessing

The preprocessing pipeline includes:

- Lowercasing tweet text
- Removing URLs
- Filtering non-English tweets
- Removing duplicate and null values
- Lemmatizing hashtags
- Removing stop words
- Dropping irrelevant columns
- Creating an hourly index column
- Encoding categorical columns
- Computing tweet frequency per user per hour
- Aggregating likes and retweets by user-hour
- Removing high-frequency outliers
- Creating BERT vectors for tweet content

For the prediction task, users are represented across hourly sequences. The first 38 hours are used as input, and the next-hour tweet frequency is used as the prediction target.

---

## 2. Exploratory Data Analysis

The project includes analyses of:

- Top hashtags by frequency
- Average followers per hashtag
- Likes and retweets received by top hashtags
- Hourly activity of highly active users
- Tweet counts by follower count range
- Word cloud of frequent non-hashtag terms

These analyses help identify popularity indicators and engagement patterns associated with hashtag propagation.

---

## 3. Network Analysis

A user-user interaction graph is created using retweet-based relationships.

### Graph Definition

- **Nodes:** Twitter/X users
- **Edges:** Retweet-based interactions between users

The graph is used to study how users interact with one another while participating in the `#Bitcoin` discussion.

### Community Detection

Communities are detected using the **Greedy Modularity Maximization** algorithm.

The resolution parameter is tuned to improve the detection of smaller communities and reduce unassigned outlier nodes.

### Community Metrics

For the detected communities, the following network metrics are calculated:

- Community size
- Number of edges
- Density
- Average path length
- Average degree
- Mean clustering coefficient

---

## 4. Community Leader Sentiment Analysis

Community leaders are identified based on graph connectivity.

A user is considered a community leader if they have a high degree within the detected community structure.

The tweets of community leaders are analyzed using an **ALBERT model trained on the SST-2 sentiment analysis dataset**.

The goal is to understand whether positive or negative sentiment from influential users is associated with higher user engagement.

---

## 5. Activity Prediction Models

The project predicts the tweet frequency of users in the next hour.

Two LSTM-based models are implemented.

---

### 5.1 Multivariate Numerical LSTM

The multivariate model uses numerical user and tweet features.

Features include:

- Retweets received
- Likes received
- Verified or non-verified status
- User followers
- User following
- Average likes
- Average retweets
- Frequency

Input shape:

```text
(users, hours, features)

## Dataset Link
https://www.kaggle.com/datasets/kumari2000/bitcoin-tweets-2022 
