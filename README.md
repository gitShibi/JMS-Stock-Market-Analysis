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
 With its interactive components and predictive capabilities, the dashboard empowers users to make data-driven investment decisions efficiently. 

### Results and Discussion
This section presents the results of the stock price forecasting models using ARIMA and LSTM across three time horizons: Short Range, Medium Range, and Long Range. The key evaluation metrics considered are Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). These metrics help assess the accuracy and reliability of each model.

1. ## Short-Range Forecasting Performance
Short-range forecasting evaluates the models' ability to predict stock prices in the near future. ## The results are as follows:
Stock |	Model |	RMSE |	MAE | 	MAPE (%)|
ABNB	LSTM	0.5448	0.3681	0.28%
	ARIMA	0.5350	0.4729	0.36%
BKNG	LSTM	16.6791	12.79	0.27%
	ARIMA	15.1332	11.0473	0.23%
EXPE	LSTM	0.7163	0.5486	0.32%
	ARIMA	0.4933	0.3661	0.22%
MAR	LSTM	0.0382	0.0259	3.16%
	ARIMA	1.2291	1.1327	0.39%
Short-Range Key Insights:
•	Lower RMSE and MAPE values indicate better model performance.
•	For ABNB, ARIMA performed slightly better in terms of RMSE (0.5350 vs. 0.5448) but had a higher MAE than LSTM.
•	For BKNG and EXPE, ARIMA outperformed LSTM with significantly lower RMSE, MAE, and MAPE values.
•	For MAR, LSTM showed better performance with the lowest RMSE (0.0382) and MAE (0.0259), although ARIMA had a lower MAPE (0.39% vs. 3.16%).
🔹 Conclusion for Short Range:
ARIMA generally provided better short-term predictions, particularly for BKNG and EXPE, where it significantly reduced the error compared to LSTM. However, LSTM performed slightly better for MAR, suggesting it may be more effective for certain stocks.

2. ## Medium-Range Forecasting Performance
Medium-range forecasting assesses the models' ability to predict stock prices over a slightly longer horizon.
Stock	Model	RMSE	MAE	MAPE (%)
ABNB	LSTM	3.4259	2.8584	2.13%
	ARIMA	1.8117	1.3461	1.01%
BKNG	LSTM	92.2659	75.3718	1.61%
	ARIMA	77.5974	67.0683	1.42%
EXPE	LSTM	7.3502	5.9608	8.09%
	ARIMA	2.4233	1.8677	1.12%
MAR	LSTM	4.4342	3.5057	1.28%
	ARIMA	2.8700	2.2488	0.82%
Medium-Range Key Insights:
•	For all stocks, ARIMA outperformed LSTM with lower RMSE, MAE, and MAPE values.
•	The largest improvement was for EXPE, where ARIMA reduced RMSE from 7.35 (LSTM) to 2.42, MAE from 5.96 to 1.86, and MAPE from 8.09% to 1.12%.
•	Similar improvements were seen in ABNB, BKNG, and MAR, where ARIMA consistently had lower error rates.
🔹 Conclusion for Medium Range:
ARIMA was the superior model for all stocks in medium-range forecasting, showing better accuracy and lower prediction errors compared to LSTM.

3. ## Long-Range Forecasting Performance
Long-range forecasting evaluates the models' predictive power over extended periods.
Stock	Model	RMSE	MAE	MAPE (%)
ABNB	LSTM	7.0142	5.3393	4.11%
	ARIMA	4.2303	3.1011	2.32%
BKNG	LSTM	374.1403	216.5242	24.52%
	ARIMA	261.7524	123.6059	16.14%
EXPE	LSTM	10.9731	8.0868	6.27%
	ARIMA	29.1734	24.1380	21.62%
MAR	LSTM	23.1361	12.4462	11.06%
	ARIMA	16.4630	6.2004	5.98%
Long-Range Key Insights:
•	ARIMA still performed better than LSTM for ABNB and BKNG, with lower RMSE, MAE, and MAPE values.
•	However, for EXPE and MAR, LSTM showed better results in RMSE and MAE, while ARIMA had higher errors.
•	The biggest gap was in BKNG, where ARIMA improved RMSE from 374.14 (LSTM) to 261.75 and reduced MAPE from 24.52% to 16.14%.
•	Despite ARIMA's lower errors, both models struggled with long-term forecasts, with error rates increasing significantly over time.
🔹 Conclusion for Long Range:
For ABNB and BKNG, ARIMA was the better choice, while EXPE and MAR had better results with LSTM. However, long-range forecasting remains challenging for both models, with errors increasing significantly compared to short- and medium-range predictions.

# Final Model Comparison and Business Insights
Overall Best Model Per Time Horizon:
•	Short Range → ARIMA (Best for BKNG & EXPE), LSTM (Best for MAR)
•	Medium Range → ARIMA outperformed LSTM for all stocks
•	Long Range → ARIMA was better for ABNB & BKNG, but LSTM worked better for EXPE & MAR
Key Business Takeaways:
•	ARIMA is the best choice for short- and medium-term predictions, offering higher accuracy and lower errors.
•	LSTM struggles in short-range and medium-range forecasting but may perform better for some long-range stock predictions.
•	Long-range forecasting remains unreliable with both models, indicating the need for alternative approaches, such as hybrid models (ARIMA + LSTM) or more advanced deep learning techniques.

# Conclusion & Future Work
•	The ARIMA model consistently outperformed LSTM in short- and medium-range predictions, making it a strong choice for stock traders focusing on short- to mid-term investments.
•	Long-range forecasting remains challenging, suggesting that future work should explore hybrid models that combine ARIMA’s strengths in short-term forecasting with LSTM’s ability to capture complex patterns.
•	Additional market indicators (e.g., macroeconomic trends, social sentiment analysis) could improve prediction accuracy.
By leveraging the best model per time horizon, investors and traders can make more informed decisions when analyzing stock price movements. 

### Conclusions and Recommendations
The analysis of stock market data using ARIMA and LSTM models across short, medium, and long-range forecasting has provided valuable insights into their respective predictive capabilities. The key conclusions from this study are as follows:
1.	Model Performance Across Different Time Horizons:
o	Short-Range Forecasting: ARIMA outperformed LSTM for most stocks, particularly in RMSE and MAPE, indicating higher accuracy in short-term price movements.
o	Medium-Range Forecasting: ARIMA continued to demonstrate superior performance, especially in predicting stock trends with lower errors across all three key metrics (RMSE, MAE, and MAPE).
o	Long-Range Forecasting: LSTM performed better for some stocks in capturing long-term trends but had higher error margins compared to ARIMA. ARIMA exhibited higher stability but struggled with certain stocks that showed high volatility (e.g., BKNG and EXPE).
2.	Stock-Specific Insights:
o	ABNB & EXPE: ARIMA consistently produced lower RMSE and MAPE values across all forecasting ranges, making it a more reliable choice.
o	BKNG: The LSTM model struggled significantly in long-term forecasts (MAPE = 24.52%), whereas ARIMA, though better, still showed high errors (MAPE = 16.14%).
o	MAR: For short-term predictions, LSTM performed better, but in medium and long-term forecasts, ARIMA had superior accuracy.
3.	General Observations:
o	ARIMA models were more effective for medium-range stock price predictions, offering reliable performance with lower error rates.
o	LSTM models showed potential in capturing non-linear trends but had difficulty maintaining accuracy over long time horizons.
o	ARIMA models were easier to interpret and required less computational power, making them suitable for practical financial decision-making.

# Limitations
Despite the promising results, the study had certain limitations:
1.	Stationarity Assumption:
o	ARIMA models require stationary data, and differencing was applied to ensure stationarity. However, this transformation might have resulted in the loss of some valuable market trend information.
2.	Limited Feature Selection:
o	The study primarily used technical indicators like moving averages (MA_10, MA_50), RSI, and volume. Including macroeconomic factors (e.g., interest rates, market indices) could improve forecasting accuracy.
3.	Handling of Market Volatility:
o	Both models struggled with high volatility stocks like BKNG, suggesting the need for hybrid models that incorporate volatility-adjusted techniques.
4.	Deep Learning Computational Costs:
o	LSTM models require significant computational resources for training and optimization, which may not be practical for all financial analysts or retail investors.

# Recommendations & Future Research Directions
1.	Hybrid Models for Enhanced Forecasting:
o	Combining ARIMA and LSTM into a hybrid ARIMA-LSTM model could leverage the strengths of both approaches: ARIMA’s ability to capture linear trends and LSTM’s ability to learn complex, non-linear relationships.
2.	Incorporation of Fundamental Analysis:
o	Future models should integrate fundamental data such as earnings reports, interest rates, economic indicators, and market sentiment analysis to improve predictive accuracy.
3.	Alternative Machine Learning Models:
o	Exploring Transformer-based models (e.g., Attention Mechanisms) or XGBoost could further enhance long-term stock predictions, especially for volatile stocks.
4.	Fine-Tuning and Hyperparameter Optimization:
o	Further optimization of LSTM’s hyperparameters (e.g., learning rate, batch size) and ARIMA’s order selection could lead to better performance.
5.	Real-Time Implementation for Trading Strategies:
o	Deploying the model in a real-time setting with live stock market data feeds could provide traders with actionable insights, allowing for automated trading and risk management strategies.

# Final Thoughts
This study demonstrated that ARIMA is a robust choice for short-to-medium-term forecasting, while LSTM has potential for long-term trends but requires further fine-tuning. Future research should focus on hybrid models, feature expansion, and real-time applications to enhance stock market prediction accuracy.
By leveraging these insights, investors and financial analysts can make more informed trading decisions and improve risk management strategies in the dynamic stock market environment.

### Project Title and Description.
Project Title: JMS-Stock-Market-Analysis Project
Project Description
This project focuses on predicting stock prices using ARIMA (Autoregressive Integrated Moving Average) and LSTM (Long Short-Term Memory) models. The analysis covers short, medium, and long-range forecasting, comparing the accuracy and performance of both models across different stock symbols (ABNB, BKNG, EXPE, MAR).
The project involves:
•	Data Preprocessing & Feature Engineering: Using moving averages, RSI, and volume as key features.
•	Time Series Forecasting: Implementing ARIMA for linear trend modeling and LSTM for deep learning-based sequence prediction.
•	Model Performance Evaluation: Comparing RMSE, MAE, and MAPE to assess forecasting accuracy.
•	Dashboard Development: (Future scope) A user-friendly interface to visualize predictions and key metrics.


### Instructions on how to run the code and use the dashboard.
1. Set Up the Environment
Make sure you have Python 3.8+ installed. Clone or download the project repository.
2. Install Required Dependencies
Run the following command to install all necessary libraries:
```!pip install yfinance
```!pip install pmdarima
``` !pip install prophet
```!pip install fbprophet

3. Prepare the Dataset
Ensure that the cleaned and preprocessed dataset is available in the /datasets/ folder in CSV format. If needed, rerun the preprocessing scripts.
4. Train the Models
Run the following script to train ARIMA and LSTM models:
 ``` Python train_models.py


• The Trained models will be saved in the /models/ directory.
• The Best ARIMA model parameters are automatically selected using Auto-ARIMA.
• The LSTM model uses hyper parameter tuning for better optimization.
5. Evaluate Model Performance
To test model accuracy and generate key performance metrics, run:
  ```Python evaluate_models.py

• This Script will display RMSE, MAE, and MAPE for short, medium, and long-range forecasts.
• a comparison table will be generated to identify the best-performing model.
6. Run the Dashboard
The dashboard can be launched using:
     ```Streamlit run dashboard.py

•	Features:
o	Interactive stock price visualization (historical and predicted).
o	Model performance comparison.
o	Adjustable forecasting horizons.
### Dependencies (Libraries Used)
The project requires the following Python libraries:
These are listed in requirements.txt, which allows easy installation.


 ```libraies imported
```import yfinance as yf
```import pandas as pd
```import matplotlib.pyplot as plt
```import seaborn as sns
```import numpy as np
```from plotly.subplots import make_subplots
```import plotly.graph_objects as go
```import plotly.express as px
```from sklearn.preprocessing import MinMaxScaler
```import torch
```import torch.nn as nn
```import torch.optim as optim
```from torch.utils.data import DataLoader, TensorDataset
```from statsmodels.tsa.stattools import adfuller
```from statsmodels.tsa.statespace.sarimax import SARIMAX  from statsmodels.tsa.arima.model import ARIMA
```from pmdarima import auto_arima

### Acknowledgments & Contributions
This project was developed as part of a stock market forecasting study. We would like to express our sincere gratitude to Dr. Menor Tekeba for his invaluable guidance and support throughout the project. We also extend our appreciation to OpenAI and Statsmodels Documentation for providing insights into ARIMA models, as well as TensorFlow and Keras Documentation for their comprehensive resources on LSTM model training. Additionally, we acknowledge the contributions of various financial data providers for supplying the stock market data that made this analysis possible.
Contributors:
•	[Jemila Muhdin, Melat Miheretab, Shibeshi Getachew] - Model development, data analysis, and report writing.











