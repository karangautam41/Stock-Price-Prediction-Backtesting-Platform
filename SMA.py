import streamlit as st
import pandas as pd
import numpy as np

# This patch is good for local robustness but pinning versions in requirements.txt is the main fix.
if not hasattr(np, 'bool_'):
    np.bool_ = bool
if not hasattr(np, 'int_'):
    np.int_ = int
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import seaborn as sns
from sklearn.metrics import confusion_matrix
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima as pm

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Stock Analysis & Trading Backtester",
    page_icon="📊",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Loads historical stock data from Yahoo Finance and ensures single-level columns."""
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        st.error(f"No data found for ticker '{ticker}' in the given date range. Please select a different stock or date range.")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [col.capitalize() for col in data.columns]
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

@st.cache_data
def calculate_technical_indicators(df):
    """Calculates various technical indicators for the stock data."""
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    epsilon = 1e-10
    rs = gain / (loss + epsilon)
    
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df.dropna()

@st.cache_data
def prepare_data_for_model(df, horizon=5):
    """Prepares features (X) and target (y) for the ML model."""
    df_model = df.copy()
    df_model.dropna(inplace=True)
    df_model['Target'] = np.where(df_model['Close'].shift(-horizon) > df_model['Close'], 1, 0)
    df_model.dropna(inplace=True)
    features = [col for col in df_model.columns if col not in ['Target', 'Open', 'High', 'Low']]
    return df_model, features

# --- Model Training (Cached) ---

@st.cache_resource
def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_lstm(_X_train_seq, _y_train_seq):
    """Trains an LSTM neural network."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(_X_train_seq.shape[1], _X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(_X_train_seq, _y_train_seq, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    return model

# --- Simple, robust backtester (replacement for external 'backtesting' dependency) ---

def run_simple_backtest(df, signal_col='Signal', cash=100_000, commission=0.002):
    """Run a straightforward buy-all / sell-all backtest using the provided signal column."""
    df = df.copy().reset_index()
    position_shares = 0
    cash = float(cash)
    equity_list = []
    trades = []
    entry_price = None

    for i, row in df.iterrows():
        price = float(row['Close'])
        sig = int(row[signal_col]) if signal_col in df.columns else 0

        # Buy signal
        if sig == 1 and position_shares == 0:
            max_shares = int((cash) // (price * (1 + commission)))
            if max_shares > 0:
                cost = max_shares * price * (1 + commission)
                cash -= cost
                position_shares = max_shares
                entry_price = price
                # FIX 1: Use the actual date from the row, not the new integer index.
                trades.append({'Date': row['Date'], 'Type': 'Buy', 'Price': price, 'Shares': max_shares})

        # Sell signal
        elif sig == 0 and position_shares > 0:
            proceeds = position_shares * price * (1 - commission)
            pnl = (price - entry_price) * position_shares if entry_price is not None else None
            cash += proceeds
             # FIX 1 (cont.): Use the actual date from the row.
            trades.append({'Date': row['Date'], 'Type': 'Sell', 'Price': price, 'Shares': position_shares, 'P/L': pnl})
            position_shares = 0
            entry_price = None

        equity = cash + position_shares * price
        equity_list.append(equity)

    # Calculate metrics
    # FIX 2: Index the equity series by the correct dates, not the new integer index.
    equity_series = pd.Series(equity_list, index=df['Date'])
    returns = equity_series.pct_change().fillna(0)
    total_return_pct = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
    
    if returns.std() == 0:
        sharpe = float('nan')
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
    cum_max = equity_series.cummax()
    drawdown = (equity_series - cum_max) / cum_max
    max_drawdown_pct = drawdown.min() * 100
    
    sells = [t for t in trades if t['Type'] == 'Sell']
    wins = sum(1 for t in sells if t.get('P/L', 0) > 0)
    win_rate = (wins / len(sells) * 100) if len(sells) > 0 else 0

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.set_index('Date')


    return {
        'equity_series': equity_series,
        'Return [%]': total_return_pct,
        'Sharpe Ratio': sharpe,
        'Max. Drawdown [%]': max_drawdown_pct,
        'Win Rate [%]': win_rate,
        'trades': trades_df
    }

# --- Streamlit UI ---
st.title('📊 Advanced Stock Analysis & Trading Backtester')
st.markdown("""
- **Tab 1:** Analyze underlying patterns in the stock price.
- **Tab 2:** Backtest a trading strategy using Random Forest or LSTM models.
- **Tab 3:** Forecast future stock prices using a classical ARIMA model.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    nifty50_tickers = {
        "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "HDFCBANK": "HDFCBANK.NS",
        "INFY": "INFY.NS", "ICICIBANK": "ICICIBANK.NS", "HINDUNILVR": "HINDUNILVR.NS",
        "KOTAKBANK": "KOTAKBANK.NS", "SBIN": "SBIN.NS", "BAJFINANCE": "BAJFINANCE.NS"
    }
    selected_stock_name = st.selectbox("Select Stock Ticker", list(nifty50_tickers.keys()))
    ticker = nifty50_tickers[selected_stock_name]
    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

# --- Data Loading and Processing ---
data = load_data(ticker, start_date, end_date)
if data is None:
    st.stop()

with st.spinner("Calculating technical indicators..."):
    data_with_indicators = calculate_technical_indicators(data)
st.success(f"Successfully loaded and processed {len(data_with_indicators)} days of data for {selected_stock_name}.")

# --- Main Page Tabs ---
tab1, tab2, tab3 = st.tabs(["🔎 EDA & Time Series Analysis", "🤖 Machine Learning Backtesting", "📈 ARIMA Forecasting"]) 

with tab1:
    st.header(f"Exploratory Data Analysis for {selected_stock_name}")
    st.subheader("Price Chart & Technical Indicators")
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(x=data_with_indicators.index, open=data_with_indicators['Open'], high=data_with_indicators['High'], low=data_with_indicators['Low'], close=data_with_indicators['Close'], name='Price'))
    fig_price.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['SMA_20'], name='SMA 20'))
    fig_price.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['SMA_50'], name='SMA 50'))
    fig_price.update_layout(title_text=f'{selected_stock_name} Candlestick Chart', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")
    st.subheader("Time Series Decomposition")
    st.write("This plot breaks down the stock's closing price into: **Trend**, **Seasonality**, and **Residuals**.")
    decomposition = seasonal_decompose(data_with_indicators['Close'], model='multiplicative', period=252) # Assuming daily data, 252 trading days in a year
    fig_decomp = decomposition.plot()
    fig_decomp.set_size_inches(12, 8)
    st.pyplot(fig_decomp)

    st.markdown("---")
    st.subheader("Stationarity Test (Augmented Dickey-Fuller)")
    st.write("We look for a **p-value < 0.05** to consider the series stationary.")
    adf_result = adfuller(data_with_indicators['Close'].dropna())
    st.write(f'ADF Statistic: **{adf_result[0]:.4f}**')
    st.write(f'p-value: **{adf_result[1]:.4f}**')
    if adf_result[1] < 0.05: st.success("The data is likely stationary.")
    else: st.warning("The data is likely non-stationary.")

    st.markdown("---")
    st.subheader("Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots")
    col1, col2 = st.columns(2)
    with col1:
        st.write("ACF")
        fig_acf = plt.figure(figsize=(8, 4))
        plot_acf(data_with_indicators['Close'], ax=plt.gca(), lags=40)
        st.pyplot(fig_acf)
    with col2:
        st.write("PACF")
        fig_pacf = plt.figure(figsize=(8, 4))
        plot_pacf(data_with_indicators['Close'], ax=plt.gca(), lags=40)
        st.pyplot(fig_pacf)

with tab2:
    st.header("Backtest Trading Strategy with Machine Learning")
    model_choice = st.radio("Choose Prediction Model", ("Random Forest", "LSTM (Deep Learning)"), key='ml_model')
    
    with st.form("ml_backtest_form"):
        st.write("Configure and run the backtest")
        initial_cash = st.number_input("Initial Cash", min_value=1000, value=100_000, step=1000)
        commission_pct = st.number_input("Commission (%)", min_value=0.0, value=0.2, step=0.01)
        run_button = st.form_submit_button("🚀 Run ML Backtest")

    if run_button:
        df_model, features = prepare_data_for_model(data_with_indicators)
        X = df_model[features]
        y = df_model['Target']
        if len(df_model) < 100:
            st.error("Not enough data for a reliable backtest. Please select a larger date range.")
        else:
            train_size = int(len(df_model) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            predictions, y_test_aligned = None, y_test
            time_steps = 30
            
            if model_choice == "Random Forest":
                with st.spinner("Training Random Forest..."):
                    model = train_random_forest(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                st.success("Random Forest model trained!")
                
            elif model_choice == "LSTM (Deep Learning)":
                with st.spinner("Training LSTM model..."):
                    def create_sequences(X, y, time_steps):
                        Xs, ys = [], []
                        for i in range(len(X) - time_steps):
                            Xs.append(X[i:(i + time_steps)])
                            ys.append(y[i + time_steps])
                        return np.array(Xs), np.array(ys)

                    if len(X_test_scaled) <= time_steps:
                        st.error(f"Test data is too short for LSTM with {time_steps} time steps.")
                    else:
                        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, time_steps)
                        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, time_steps)
                        model = train_lstm(X_train_seq, y_train_seq)
                        lstm_preds_proba = model.predict(X_test_seq)
                        predictions = (lstm_preds_proba > 0.5).astype(int).flatten()
                        y_test_aligned = pd.Series(y_test_seq, index=X_test.index[time_steps:])
                        st.success("LSTM model trained!")

            if predictions is not None and len(predictions) > 0:
                backtest_data_with_signals = data_with_indicators.loc[y_test_aligned.index].copy()
                backtest_data_with_signals['Signal'] = predictions

                st.subheader("Model Performance on Test Data")
                st.dataframe(pd.DataFrame(classification_report(y_test_aligned, predictions, output_dict=True, zero_division=0)).transpose())
                
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_test_aligned, predictions)
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Hold (0)', 'Buy (1)'], yticklabels=['Hold (0)', 'Buy (1)'])
                plt.ylabel('Actual'); plt.xlabel('Predicted')
                st.pyplot(fig)

                with st.spinner("Running simplified backtest..."):
                    bt_results = run_simple_backtest(backtest_data_with_signals, signal_col='Signal', cash=initial_cash, commission=commission_pct / 100)

                st.subheader("📈 Backtesting Results")
                cols = st.columns(4)
                cols[0].metric("Return [%]", f"{bt_results['Return [%]']:.2f}")
                sharpe_val = bt_results['Sharpe Ratio']
                cols[1].metric("Sharpe Ratio", f"{sharpe_val:.2f}" if not np.isnan(sharpe_val) else "N/A")
                cols[2].metric("Max. Drawdown [%]", f"{bt_results['Max. Drawdown [%]']:.2f}")
                cols[3].metric("Win Rate [%]", f"{bt_results['Win Rate [%]']:.2f}")

                st.write("#### Equity Curve")
                fig_eq = go.Figure(go.Scatter(x=bt_results['equity_series'].index, y=bt_results['equity_series'].values, mode='lines', name='Portfolio Value'))
                fig_eq.update_layout(title='Portfolio Equity Over Time', xaxis_title='Date', yaxis_title='Portfolio Value ($)', height=500)
                st.plotly_chart(fig_eq, use_container_width=True)

                st.write("#### Trade History")
                if not bt_results['trades'].empty:
                    st.dataframe(bt_results['trades'])
                else:
                    st.info("No trades were executed during this backtest.")

with tab3:
    st.header("Forecast Future Prices with Auto-ARIMA")
    st.write("This tool uses an **Auto-ARIMA** model to find the best statistical fit for the historical data and forecast future prices.")

    forecast_days = st.slider("Number of days to forecast:", 5, 90, 30, key='forecast_days')

    if st.button("🔮 Generate ARIMA Forecast", key='run_arima'):
        with st.spinner("Finding best ARIMA model and generating forecast..."):
            auto_arima_model = pm.auto_arima(
                data_with_indicators['Close'],
                start_p=1, start_q=1,
                test='adf',
                max_p=5, max_q=5,
                m=1,
                d=None,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            st.info(f"Best model found by Auto-ARIMA: **ARIMA{auto_arima_model.order}**")
            forecast_values, conf_int = auto_arima_model.predict(n_periods=forecast_days, return_conf_int=True)
            forecast_index = pd.date_range(start=data_with_indicators.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_series = pd.Series(forecast_values, index=forecast_index)
            conf_int_df = pd.DataFrame(conf_int, index=forecast_index, columns=['lower', 'upper'])

            st.subheader(f"Price Forecast for the Next {forecast_days} Days")
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=data_with_indicators.index[-200:], y=data_with_indicators['Close'][-200:], mode='lines', name='Historical Close'))
            fig_forecast.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines', name='Forecast'))
            fig_forecast.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], mode='lines', line=dict(width=0), showlegend=False, name='Upper Bound'))
            fig_forecast.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], mode='lines', line=dict(width=0), fill='tonexty', name='95% Confidence Interval'))
            fig_forecast.update_layout(title_text=f'Auto-ARIMA Forecast for {selected_stock_name}', yaxis_title='Price')
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.subheader("Forecasted Values")
            forecast_display_df = pd.DataFrame({'Forecast': forecast_series, 'Lower Bound (95%)': conf_int_df['lower'], 'Upper Bound (95%)': conf_int_df['upper']})
            st.dataframe(forecast_display_df)

            with st.expander("Show Model Diagnostics"):
                fig_diag = auto_arima_model.plot_diagnostics(figsize=(15, 12))
                st.pyplot(fig_diag)