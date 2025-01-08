import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    """Fetch IBOVESPA data for the last 20 years"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20*365)  # 20 years
    ibov = yf.download('^BVSP', start=start_date, end=end_date)
    return ibov

def calculate_returns(data):
    """Calculate log returns"""
    return np.log(data['Close']/data['Close'].shift(1)).dropna()

def find_best_model(returns):
    """Find best EGARCH model based on AIC"""
    best_aic = np.inf
    best_order = None
    
    for p in range(1, 3):  # ARCH order
        for q in range(1, 3):  # GARCH order
            try:
                model = arch_model(returns, p=p, q=q, vol='EGARCH', dist='normal')
                results = model.fit(disp='off')
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, q)
            except:
                continue
    
    return best_order

def monte_carlo_simulation(model_fit, n_simulations=10000, forecast_period=21):
    """Perform Monte Carlo simulation for future returns"""
    params = model_fit.params
    mu = params['mu']  # mean return
    omega = params['omega']
    alpha1 = params['alpha[1]']
    alpha2 = params['alpha[2]']
    beta1 = params['beta[1]']
    beta2 = params['beta[2]']
    
    # Initialize arrays
    simulated_returns = np.zeros((n_simulations, forecast_period))
    
    # Get last volatility
    last_vol = model_fit.conditional_volatility[-1]
    
    # Calculate annualized volatility for scaling
    annual_vol = np.sqrt(252) * last_vol
    
    for sim in range(n_simulations):
        # Initialize volatility
        log_sigma2 = np.log(last_vol**2)
        
        for t in range(forecast_period):
            # Generate random shock
            z = np.random.standard_normal()
            
            # Update log-variance (EGARCH)
            log_sigma2 = omega + alpha1 * (abs(z) - np.sqrt(2/np.pi)) + alpha2 * (abs(z) - np.sqrt(2/np.pi)) + \
                        (beta1 + beta2) * log_sigma2
            
            # Calculate volatility
            sigma = np.sqrt(np.exp(log_sigma2))
            
            # Generate return with current volatility and drift adjustment
            r = mu - 0.5 * sigma**2 + sigma * z  # Include volatility drift adjustment
            simulated_returns[sim, t] = r
    
    return simulated_returns

def main():
    # Fetch data
    print("Fetching IBOVESPA data...")
    ibov = fetch_data()
    returns = calculate_returns(ibov)
    
    # Find best model
    print("Finding best EGARCH model...")
    best_order = find_best_model(returns)
    p, q = best_order
    
    # Fit model
    print(f"Fitting EGARCH({p},{q}) model...")
    model = arch_model(returns, p=p, q=q, vol='EGARCH', dist='normal')
    model_fit = model.fit(disp='off')
    
    # Print model parameters
    print("\nModel parameters:")
    print(model_fit.params)
    
    # Monte Carlo simulation
    print("\nPerforming Monte Carlo simulation...")
    simulated_returns = monte_carlo_simulation(model_fit)
    
    # Calculate cumulative returns
    cumulative_returns = np.sum(simulated_returns, axis=1)
    
    # Calculate price predictions
    last_price = float(ibov['Close'].iloc[-1])  # Convert to float
    predicted_prices = last_price * np.exp(cumulative_returns)
    
    # Calculate confidence intervals
    confidence_level = 0.99
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    lower_bound = np.percentile(predicted_prices, lower_percentile * 100)
    upper_bound = np.percentile(predicted_prices, upper_percentile * 100)
    
    # Print results
    print("\nResults:")
    print(f"Current IBOVESPA: {last_price:.2f}")
    print(f"99% Confidence Interval for 21 days ahead:")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_prices, bins=50, density=True, alpha=0.7)
    plt.axvline(lower_bound, color='r', linestyle='--', label='99% CI')
    plt.axvline(upper_bound, color='r', linestyle='--')
    plt.axvline(last_price, color='g', linestyle='-', label='Current Price')
    plt.title('Distribution of Predicted IBOVESPA Values (21 days ahead)')
    plt.xlabel('IBOVESPA')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('ibovespa_prediction.png')
    plt.close()

if __name__ == "__main__":
    main()
