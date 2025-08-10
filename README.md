

# Time Series Forecasting for Portfolio Management Optimization

This project develops a data-driven framework to enhance portfolio management strategies by applying time series forecasting to historical financial data. It uses statistical and deep learning models to predict stock price movements, which inform portfolio optimization and risk management decisions.

## Table of Contents

  - [Business Objective](https://www.google.com/search?q=%23business-objective)
  - [Features](https://www.google.com/search?q=%23features)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Technologies Used](https://www.google.com/search?q=%23technologies-used)
  - [Installation and Usage](https://www.google.com/search?q=%23installation-and-usage)
  - [Results](https://www.google.com/search?q=%23results)
  - [Future Work](https://www.google.com/search?q=%23future-work)
  - [License](https://www.google.com/search?q=%23license)
  - [Contact](https://www.google.com/search?q=%23contact)

-----

## Business Objective

The project is framed from the perspective of a Financial Analyst at **Guide Me in Finance (GMF) Investments**, a financial advisory firm. GMF aims to leverage advanced time series forecasting models to predict market trends, optimize asset allocation, and enhance portfolio performance.The primary goal is to help clients achieve their financial objectives by minimizing risks and capitalizing on market opportunities.This involves analyzing key assets like TSLA, SPY, and BND to provide data-driven investment recommendations

-----

## Features

  * **Data Extraction & Cleaning:** Fetches historical financial data from the YFinance API and cleans it by handling non-trading days (weekends, holidays).
  * **Exploratory Data Analysis (EDA):** Provides in-depth visual and statistical analysis of price trends, daily returns, and volatility for TSLA, SPY, and BND.
  * **Risk Metric Calculation:** Computes foundational risk metrics like Value at Risk (VaR) and the Sharpe Ratio.
  * **Comparative Modeling:** Implements, trains, and evaluates two different forecasting models for TSLA's stock price:
      * A classical statistical model (**ARIMA**)
      * A deep learning model (**LSTM**)
  * **Modular Codebase:** The entire workflow is organized into reusable functions and clear, narrative notebooks.

-----

## Project Structure

The project is organized into a modular structure to separate concerns, making it clean, scalable, and easy to navigate.

```
Time-Series-Forecasting/
├── .venv/                  # Virtual environment files
├── data/                   # (Optional) For storing raw/processed data files
├── models/                 # Saved trained models (e.g., tsla_lstm_model.keras)
├── notebooks/              # Jupyter notebooks for analysis and reporting
│   ├── clean_and_EDA.ipynb
│   ├── extraction.ipynb
│   └── modeling.ipynb
├── src/                    # Source code with modular Python functions
│   ├── yf_clean.py         # Functions for data cleaning
│   ├── yf_extract.py       # Functions for data extraction
│   └── yf_tesla_modeling.py # Functions for ARIMA/LSTM modeling & evaluation
├── .gitignore
├── LICENSE
├── README.md               # This file
└── requirements.txt        # Project dependencies
```

-----

## Technologies Used

  * **Programming Language:** Python 3.11
  * **Data Handling & Analysis:** Pandas, NumPy
  * **Data Extraction:** YFinance
  * **Statistical Modeling:** Statsmodels, Pmdarima
  * **Deep Learning:** TensorFlow, Keras
  * **Data Scaling:** Scikit-learn
  * **Visualization:** Matplotlib, Plotly

-----

## Installation and Usage

Follow these steps to set up and run the project locally.

**1. Clone the repository:**

```bash
git clone https://github.com/sumeyaaaa/Time-Series-Forecasting
cd Time-Series-Forecasting
```

**2. Create and activate a virtual environment:**

```bash
# Create the environment
python -m venv .venv

# Activate the environment (Windows)
.\.venv\Scripts\Activate.ps1

# Activate the environment (Mac/Linux)
source .venv/bin/activate
```

**3. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

**4. Run the analysis:**
The analysis is best viewed by running the Jupyter Notebooks in the `notebooks/` directory in the following order:

1.  `extraction.ipynb`
2.  `clean_and_EDA.ipynb`
3.  `modeling.ipynb`

-----

## Results

The primary outcome of the initial analysis was the successful development and comparison of ARIMA and LSTM models to forecast TSLA's stock price. The LSTM model significantly outperformed the simpler ARIMA model, demonstrating its superior ability to capture the complex patterns in the financial data.

| Metric | ARIMA Model | **LSTM Model** |
| :--- | :--- | :--- |
| MAE | 63.71 | **12.58** |
| RMSE | 78.99 | **17.57** |
| MAPE | 24.15% | **4.46%** |

This confirms that the deep learning approach provides a more accurate forecast for the next stages of the project.

-----

## Future Work

The project is ongoing. The next steps will focus on using the insights from the successful LSTM model to perform portfolio management tasks:

  * **Task 3: Future Forecasting:** Use the trained LSTM to forecast 6-12 months into the future with confidence intervals.
  * **Task 4: Portfolio Optimization:** Apply Modern Portfolio Theory (MPT) to find the optimal asset weights for a portfolio containing TSLA, SPY, and BND.
  * **Task 5: Strategy Backtesting:** Simulate the performance of the optimized portfolio against a benchmark to validate the strategy.

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----

## Contact

Your Name - [your.sumeyasirmulach@gmail.com]

Project Link: [https://github.com/sumeyaaaa/Time-Series-Forecasting]