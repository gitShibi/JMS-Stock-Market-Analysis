# JMS-Stock-Market-Analysis Project
### Executive Summary
This report presents a comprehensive stock market analysis focusing on four selected stocks: Airbnb (ABNB), Booking Holdings (BKNG), Expedia (EXPE), and Marriott International (MAR), the data collection each stock symbol uses has three parameters that comprises Short Range data for a 1 month period with time interval of 2 minutes, Medium Range that uses data for 2 years period with a time interval of 1 hour and Long Range data for a period of “max” with a time interval of 1day.   The project aims to analyze the above mentioned three time range stock trends for four selected companies using historical data from Yahoo Finance. The methodology includes data cleaning, exploratory data analysis (EDA), predictive modeling using machine learning techniques (ARIMA and LSTM), and dashboard development for interactive visualization. Key findings reveal trends in stock performance, correlations among stocks, and the effectiveness of different predictive models such as ARIMA (Autoregressive Integrated Moving Average) and LSTM (Long Short-Term Memory) for forecasting stock prices, with LSTM demonstrating superior accuracy in forecasting non-linear stock price movements. Future enhancements could include integrating external economic indicators and refining model architectures for improved predictive performance.

### Introduction
The Project is a Stock Market Data Analysis, Visualization and Best Predictive Models Development. The objective of this project is to analyze Short Range, Medium-Range and Long Range stock trends and develop predictive models for informed investment decisions. The selected stocks for this project are Airbnb (ABNB), Booking Holdings (BKNG), Expedia (EXPE), and Marriott International (MAR). The analysis involves collecting historical stock data for the above mentioned stocks, performing EDA, and building predictive models to forecast future price movements.

### Data Collection and Preparation
Historical stock data was sourced from Yahoo Finance. The dataset was cleaned by handling missing values, standardizing date formats, and ensuring consistency in stock attributes such as open, close, high, low, and volume.

### Exploratory Data Analysis (EDA)
EDA was conducted using visualizations and statistical analysis. Key insights include:
•	Trends and seasonal patterns in stock prices using line charts and box plots.
  
       

•	Calculate Summary Statistics:      ABNB has a moderate average closing price with relatively low volatility compared to the other stocks.
BKNG has the highest average closing price, along with significant volatility (as indicated by the high standard deviation), which could be attributed to large fluctuations in the stock price.
EXPE is positioned in between ABNB and BKNG, with moderate average pricing and volatility.
MAR shows a slightly higher average price than ABNB, with relatively moderate volatility.
•	Correlation analysis between stocks.
•	Distribution and volatility of stock returns.
•	Moving averages and trend indicators.

### Predictive Modeling
This predictive modeling process focuses on forecasting the Short Range,medium-range and Large range stock prices for Airbnb (ABNB), Booking Holdings (BKNG), Expedia (EXPE), and Marriott (MAR) using the LSTM (Long Short-Term Memory) neural network and ARIMA(Autoregressive Integrated Moving Average) The models were trained on historical price data, and their performance was evaluated using RMSE and R-squared metrics plus hyper parameter tuning was conducted to optimize predictions. The goal is to build a reliable model that captures stock price trends using historical data, technical indicators, and deep learning.
1.	LSTM: Long Short-Term Memory (for deep learning
Feature Engineering and Data Preprocessing
To enhance model performance, key technical indicators were calculated and added as features:
•	Moving Averages (MA):
o	10-day Moving Average (MA_10)
o	50-day Moving Average (MA_50)
•	Relative Strength Index (RSI):
o	Measures stock momentum over a 14-day period
•	Volume was also considered as a feature.
The data set underwent preprocessing:
✔ Handling Missing & Infinite Values: NaNs were replaced with column means.
✔ Normalization: MinMaxScaler was used to scale data between 0 and 1 for better LSTM performance.
✔ Sequence Creation: The past 60 days' data was used to predict the next day's stock price.

LSTM Model Selection and Training
The predictive model is based on an LSTM network, chosen for its ability to capture temporal dependencies in sequential data.

Model Architecture
•	Input Size: 5 (Close Price, MA_10, MA_50, RSI, Volume)
•	Hidden Layers: 2 LSTM layers with 64 units each
•	Dropout: 20% to prevent overfitting
•	Fully Connected (FC) Layer: Outputs a single predicted stock price
Training Process
•	Loss Function: Mean Squared Error (MSE) to measure prediction accuracy
•	Optimizer: Adam Optimizer with a learning rate of 0.001
•	Epochs: 50 iterations
During training, the model minimized loss by updating weights using backpropagation.
Model Evaluation and Results
The model was tested on unseen data (20% of the dataset). The performance was evaluated using:
•	Root Mean Squared Error (RMSE): Measures absolute error in stock price predictions.
•	Mean Absolute Error (MAE): Measures the average magnitude of prediction errors.
•	Mean Absolute Percentage Error (MAPE): Indicates how much percentage error exists in predictions.
 
                                                                     Figure 5 Actual vs Predicted price with LSTM
Findings
✔ Predicted vs. Actual Prices: The LSTM model was able to follow the actual price trend with reasonable accuracy.
✔ RMSE, MAE, and MAPE Results: Provided insight into prediction errors, highlighting areas for improvement.
✔ MAPE: A lower percentage means better accuracy in forecasting.

The LSTM-based model successfully captured All-range stock price trends, leveraging historical data and technical indicators. While the results show promising accuracy, further optimizations through hyper parameter tuning and feature engineering can improve prediction robustness. This model serves as a strong foundation for developing more advanced financial forecasting tools.


Fine-Tuning & Future Enhancements
To improve the model's performance:
🔹 Hyperparameter Tuning: Adjusting the number of LSTM layers, units, and learning rate.
🔹 Alternative Loss Functions: Testing Huber Loss instead of MSE for robustness.
🔹 Feature Engineering: Incorporating additional indicators like Bollinger Bands or MACD.
🔹 Hybrid Models: Combining LSTM with ARIMA for enhanced predictive power.

2.	ARIMA: Autoregressive Integrated Moving Average
Feature Engineering and Data Preprocessing
To ensure the dataset is suitable for time series forecasting, the following preprocessing steps were performed:
Feature Selection
The dataset includes the target variable (Close price) and four technical indicators as features:
•	10-day Moving Average (MA_10)
•	50-day Moving Average (MA_50)
•	Relative Strength Index (RSI)
•	Trading Volume
Stationarity Check & Differencing
Time series models require stationarity. The Augmented Dickey-Fuller (ADF) Test was used to assess stationarity:
✔ Null Hypothesis (H0): The series is non-stationary
✔ Alternative Hypothesis (H1): The series is stationary
Initially, the test indicated non-stationarity, so first-order differencing was applied (df.diff()), making the data stationary. The ADF test was repeated after differencing to confirm stationarity.


Model Selection & Training
1. Initial ARIMAX Model
An ARIMAX model was implemented using the SARIMAX function from statsmodels. The model's components include:
•	(p, d, q) = (1,1,1):
o	p (Autoregressive Order): 1
o	d (Differencing Order): 1 (to handle stationarity)
o	q (Moving Average Order): 1
•	Seasonal Order (0,0,0,0): No seasonality in the model.
•	Exogenous Variables (X_train): The technical indicators were included to enhance predictions.
The model was trained on 80% of the dataset, while 20% was used for testing.
2. Model Evaluation
The initial ARIMAX model was evaluated using:
•	Root Mean Squared Error (RMSE)
•	Mean Absolute Error (MAE)
•	Mean Absolute Percentage Error (MAPE)
The predicted stock prices followed the general trend of actual prices but required fine-tuning.
Hyperparameter Tuning: Optimizing ARIMAX Model
To improve performance, the Auto ARIMA function (auto_arima) was used to determine the optimal (p, d, q) parameters.
✔ Best Parameters Found: (p, d, q) = (3,2,1)
A final ARIMAX model was trained using these optimized parameters:
final_model = SARIMAX(y_train, exog=X_train, order=(3,2,1), seasonal_order=(0,0,0,0)) final_model_fit = final_model.fit()
Final Model Evaluation & Results
The final model was tested on unseen data, and its performance metrics were recalculated:
📌 RMSE: Lower RMSE indicates reduced error in stock price predictions.
📌 MAE: Measures the average absolute error in predictions.
📌 MAPE: Expresses error as a percentage—lower values indicate better accuracy.
✔ The Optimized ARIMAX model showed improved accuracy compared to the initial model.
Visualization & Insights
A comparison plot of actual vs. predicted stock prices was created:
🔹 Blue Line: Actual prices
🔹 Red Line: Predicted prices
 
                                                             Figure 6 Actual vs Predicted price with ARIMAX
The final model captured stock price trends more accurately than the initial model, but some deviations remained.
Conclusion & Future Improvements
The ARIMAX model effectively leveraged exogenous variables to enhance stock price predictions. However, further refinements can be made:
🔹 Feature Engineering: Adding more indicators like MACD or Bollinger Bands.
🔹 Model Tuning: Testing seasonal ARIMA (SARIMA) if seasonality exists.
🔹 Hybrid Models: Combining ARIMAX with LSTM for improved performance.
This project demonstrated the power of ARIMAX in financial forecasting, offering valuable insights for medium-range stock price predictions.

 ### Dashboard Development
Interactive Stock Forecasting Dashboard
Introduction
This report outlines the design, functionality, and key features of an interactive Stock Forecasting Dashboard built to visualize and analyze historical and predicted stock prices for Airbnb (ABNB), Booking Holdings (BKNG), Expedia (EXPE), and Marriott International (MAR). The dashboard integrates interactive charts, key financial metrics, model comparisons, and user controls to enhance usability and decision-making.

Dashboard Design & Functionality
The dashboard is designed with a user-friendly interface, allowing traders and analysts to explore stock trends, assess model performance, and fine-tune predictions dynamically. The layout is structured into four main components:
1. Price Charts: Historical & Forecasted Prices
📌 Feature: An interactive line chart displaying historical stock prices alongside ARIMAX model forecasts.
📌 Functionality:
✔ Users can toggle between different time frames (e.g., 1 month, 3 months, 6 months, 1 year).
✔ Allows zooming & panning for detailed trend analysis.
✔ Forecasted stock prices are displayed with a confidence interval to indicate uncertainty levels.
🔹 Technology Used: Plotly, Matplotlib, or Dash for interactive visualization.

2. Key Metrics: Stock Performance Indicators
📌 Feature: A dashboard panel displaying key financial indicators in real time.
📌 Functionality:
✔ Daily Returns: Percentage change in stock price over the last trading session.
✔ Volatility: Measures the stock’s price fluctuation over time.
✔ Trading Volume: Displays the total number of shares traded within a selected time period.
✔ Moving Averages (MA_10, MA_50): Helps analyze stock momentum.
🔹 Technology Used: Dash, Streamlit

3. Model Comparisons: ARIMAX vs. Other Models
📌 Feature: A comparison panel showcasing different predictive models side by side.
📌 Functionality:
✔ Displays actual vs. predicted stock prices for multiple models (e.g., ARIMAX vs. LSTM vs. XGBoost).
✔ Includes a performance metrics table with RMSE, MAE, and MAPE for each model.
✔ Provides a heatmap of error distribution, highlighting prediction accuracy.
🔹 Technology Used: Seaborn for heatmaps, Plotly for multi-model comparison plots.

4. Interactive Controls: Customization & Exploration
📌 Feature: A control panel allowing users to customize dashboard settings dynamically.
📌 Functionality:
✔ Stock Selection: Users can switch between different assets (e.g., ABNB, BKNG,EXPE,MAR).
✔ Forecasting Horizon: Allows selection of short-term (7 days), medium-term (30 days), or long-term (90 days) predictions.
✔ Model Parameter Adjustments: Users can fine-tune ARIMAX parameters (p, d, q) and test different configurations.
✔ Toggle Indicators: Users can enable/disable technical indicators like Moving Averages, RSI, and Volume on charts.
🔹 Technology Used: Dash, Streamlit, or Voila for interactive UI elements.

Conclusion & Future Enhancements
The Stock Forecasting Dashboard provides an interactive and insightful interface for stock price analysis and prediction. Future improvements may include:
✔ Integration with Live Market Data for real-time stock updates.
✔ Additional Machine Learning Models (LSTM, Prophet) for model comparison.
✔ Automated Alerts & Notifications based on stock trend analysis.
🚀 With its interactive components and predictive capabilities, the dashboard empowers users to make data-driven investment decisions efficiently. 

### Results and Discussion






