# Development Log

Follow the analysis and human interpretation of this model at the LynxAI Blog here: https://substack.com/@lynxai

## 25/4/2026 - Day 1

Project initialized. Set up Python, VS Code, and GitHub repository.
Goal: Build an AI stock analyst that scans the S&P 500 and surfaces 
the highest-confidence short-term buy signals using machine learning.

Starting from zero Python experience. First milestone is getting 
real stock data downloaded and readable.

## 15/6/2026 - Devlopment Update
Over the past weeks, LynxAI has evolved into a functioning system. 
We have successfully 
1. Downloaded and processed 10 years of historical Apple stock data through Yahoo Finance
2. Built a data pipeline using Pandas to organize market data
3. Implemented technical indicators including RSI, MACD, 50 day moving average, 200 day moving averages, and volume trends
4. Trained our first Random Forest machine learning model
5. Generated the first live BUY / NO BUY signals and confidence scores