# CS 6120 Semester Project

This repo represents the semester project work, Development and Evaluation of Models for Coding Injury Cause Description for CS 6120.

**Name:** Lillith Chute, Vitaliy Shkolnik, Adithya Abhishek Chenthilkannan

**Email:** chute.l@northeastern.edu, Shkolnik.v@northeastern.edu, chenthilkannan.a@northeastern.edu

**Preferred Name:** Lillith

**Updated:** December 11, 2022


### About/Overview

The worker’s compensation industry contains a lot of challenges. One such challenge is finding an effective way to classify various forms of unstructured data. 
One such form deals with injury description classification. This project is designed to see if a model can be produced to effectively categorize injury 
cause descriptions using real world worker’s compensation injury description claims. Frequently these descriptions are brief, and categories are non-standardized. 
As such, analyses were done around the length of text and its nature and examination of the categories. Based on this work, it was determined that there 
is a lot of overlap in categories. Further, that the descriptions were extremely short, frequently less than the size of a tweet. Given this, we chose to 
explore three different models. Naïve Bayes was chosen as a baseline. Second, that baseline was compared against Support Vector Machine and Random Forest/XGBoost. 
It was determined that XGBoost outperformed the others. However, there was not a statistically significant difference in the performance. Based on the findings, 
a major factor in the model’s ability to predict is that the data needs to be significantly improved. Ideally, the data ingestion process could be changed such 
that overlapping categories would be combined. Several categories that have little to no data could be removed. Furthermore, descriptions would be constructed 
so that based on categories, key words would be included.
