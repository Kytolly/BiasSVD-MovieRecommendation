# Movie Recommendation Based on BiasSVD(UESTC-信息检索-期末设计) 

## Introduction

This project implements movie rating prediction based on matrix factorization algorithms. The MovieLens dataset contains rating data from multiple users for various movies, alongside movie metadata and user attributes. This dataset is commonly used for testing recommendation systems and machine learning algorithms.

For the final project, we primarily use the ratings.csv and movies.csv files from the MovieLens Latest Datasets (ml-latest.zip), which include user IDs, movie IDs, ratings, movie titles, and genres. If the hardware conditions are insufficient, the smaller dataset ml-latest-small.zip can also be used.

[The dataset can be downloaded](https://grouplens.org/datasets/movielens/).

> Dataset file descriptions:
> movies: Over 90,000 lines (movieId, title, genres)
> links: Over 90,000 lines (movieId, imdbId, tmdbId)
> ratings: Over 27,000,000 lines (userId, movieId, rating, timestamp)
> tags: Over 2,000,000 lines (userId, movieId, tag, timestamp)

## Dependency

The primary Python libraries required for this project are numpy, pandas, sklearn, and tensorflow. Refer to the requirement.txt file for specific version requirements.
For convenience, you can use the pre-configured Docker image for environment setup: docker pull MovieRec.
This project supports Windows and Linux systems, and Python version 3 or higher.

## Usage

1. Data Preprocessing:
    The dataset is split using an 8:1:1 ratio for training, validation, and testing.
    Model Training and Packaging:

2. In the dev branch, the model has been further encapsulated to support saving after training, allowing for subsequent retraining or predictions.

## Evaluation 

The rating prediction model is evaluated using Mean Squared Error (MSE). The current MSE is 1.954, indicating that the model performs well in predicting ratings for movies that users have not yet watched. Detailed training logs and evaluation information can be found in the log/info.log file.

## License
This project is licensed under the MIT License, meaning you're free to use the source code however you like.

Warning: Don’t cheat on your homework! (just for uestcer).

## Remark
For more details about the project functionality, please refer to Equation/report/report.md. The project is completed, so it may not receive further maintenance. If you have any questions or encounter any terrifying bugs, feel free to submit an issue with details, or just email me at kytolly.xqy@gmail.com. 

## Reference

[知乎-推荐基础算法之矩阵分解MF，全文可阅读；](https://zhuanlan.zhihu.com/p/268079100): An introduction to matrix factorization algorithms commonly used in recommendation systems and their applications, which is worth reading in full.