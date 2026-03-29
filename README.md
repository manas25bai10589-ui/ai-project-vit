NexTrend: Inventory & Demand Forecasting Engine
NexTrend is a predictive analytics engine built to solve the two biggest headaches in e-commerce: Stock-outs (losing sales) and Overstocking (wasted capital). By applying time-series regression to historical data, it provides actionable insights for supply chain optimization.

Note: This version of NexTrend utilizes Yahoo Finance API data as a high-volatility proxy to demonstrate how the engine handles market fluctuations and trend shifts in a real-world sales environment.

🚀 Overview
NexTrend transforms raw historical demand signals into a roadmap for future inventory requirements. It moves beyond simple "average-based" restocking by identifying the underlying momentum of product movement.

Input: Historical sales volume and price volatility.

Engine: Time-Series Analysis & Linear Regression.

Output: Predictive inventory requirements for the 30, 60, and 90-day windows.

✨ Key Features
🔍 Trend Detection: Automatically classifies product movement as Growing, Stable, or Shrinking to help managers prioritize high-velocity items.

🔮 Forecast Modeling: Uses Scikit-Learn to extrapolate patterns, accounting for noise and outliers in the data.

📊 Interactive Dashboard: A sleek Streamlit interface that allows supply chain managers to visualize "Confidence Intervals" and "Projected Stock Troughs."

🛡️ Role-Based Simulation: Designed with the workflow of a Supply Chain Manager in mind, focusing on risk mitigation.
