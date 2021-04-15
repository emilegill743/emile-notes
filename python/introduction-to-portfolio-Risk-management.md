<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Introduction to Portfolio Risk Management in Python

- [Introduction to Portfolio Risk Management in Python](#introduction-to-portfolio-risk-management-in-python)
  - [Univariate Investment Risk and Returns](#univariate-investment-risk-and-returns)
    - [Investment Risk](#investment-risk)
    - [Financial Returns](#financial-returns)
    - [Mean, Variance and Normal Distribution](#mean-variance-and-normal-distribution)
    - [Skewness and Kurtosis](#skewness-and-kurtosis)
    - [Testing for Normality - Shapiro Wilk test](#testing-for-normality---shapiro-wilk-test)
  - [Portfolio Investing](#portfolio-investing)
    - [Portfolio composition and backtesting](#portfolio-composition-and-backtesting)
    - [Correlation and co-variance](#correlation-and-co-variance)
    - [Markowitz Portfolios](#markowitz-portfolios)
  - [Factor Investing](#factor-investing)
    - [Capital Asset Pricing Model (CAPM)](#capital-asset-pricing-model-capm)
    - [Alpha and multi-factor models](#alpha-and-multi-factor-models)
    - [Expanding the 3-factor model](#expanding-the-3-factor-model)
  - [Value at Risk](#value-at-risk)
    - [Estimating Tail Risk](#estimating-tail-risk)
    - [Var Extensions](#var-extensions)
    - [Random Walks](#random-walks)

## Univariate Investment Risk and Returns

### Investment Risk

**What is risk?**
- Risk in financial markets is a measure of uncertainty.
- Dispersion or variance of financial returns.

**How is risk typically measured?**
- Standard deviation or variance of daily returns
- Kurtosis of daily returns distribution
- Skewness of daily returns distribution
- Historical drawdown

### Financial Returns
- Returns are derived from stock prices
- **Discrete returns** (simple returns) are the most commonly used, and represent periodic (daily, weekly, monthly etc.) price movements.
- **Log returns** are often used in academic research and financial modelling. They assume continuous compounding. Log returns are always smaller than discrete returns.

**Calculating returns:**

Discrete returns are calculated as the percentage change in price across a period.

$R_{t_2} = \frac{(P_{t_2} - P_{t_1})}{P_{t_2}}$

Log returns are calculated as the difference between the log of two prices.

$R_l = \ln(\frac{P_{t_2}}{P_{t_1}}) = \ln(P_{t_2}) - \ln(P_{t_1})$

Log returns aggregate across time, while discrete returns aggregate across assets.

**Calculating stock returns in Python:**

1) Load in stock prices data and store as pandas DataFrame.

```python
import pandas as pd
StockPrices = pd.read_csv('StockData.csv', parse_dates=['Date'])
StockPrices = StockPrices.sort_values(by='Date')
StockPrices.set_index('Date', inplace=True)
```

2) Calculate daiy returns of adjusted close prices and append as a new column.

```python
StockPrices["Returns"] = StockPrices["Adj Close"].pct_change()
```

Adjusted close - Normalised for stock splits, dividends and other corporate actions, giving a true reflection of the return of the stock over time.

**Visualising returns distribution:**

```python
import matplotlib.pyplot as plt

plt.hist(StockPrices["Returns"].dropna(),
         bins=75, density=False)
plt.show()
```

### Mean, Variance and Normal Distribution

Probability distributions have the following moments:

1) Mean ($\mu$)
2) Variance ($\sigma$<sup>2</sup>)
3) Skewness - measure of the tilt of a distribution
4) Kurtosis - measure of the thickness of the tails of a distribution

There are many types of distribution. Some are normal and some are non-normal. A random variable with a **Gaussian distribution** is said to be *normally distributed*.

Normal distributions have the following properties:
- Mean = $\mu$
- Variance = $\sigma$<sup>2</sup>
- Skewness = 0
- Kurtosis = 3

The standard normal distribution is a special case of the normal distribution where $\sigma = 1$ and $\mu = 0$.

Probability density function equation for a standard normal distribution:
$\frac{1}{\sqrt{2\pi\sigma^2}}e^-\frac{(x-\mu)^2}{2\sigma^2}$

**Comparing financial return distributions to the normal distribution:**

Normal distributions tend to have a skewness near 0 and kurtosis near 3. Financial returns tend not to be normally distributed and tend to have positive skewness and a kurtosis greater than 3. This implies that financial returns have a higher probability of both outliers and higher returns when compared to a normal distribution.

*Calculating mean returns in Python:*

$Average\:Annualised\:Return = ((1+\mu)^{252})-1$

where $\mu$ is the average daily return.

```python
import numpy as np

# Average daily return
np.mean(StockPrices["Returns"])

# Average annualised returns, assuming 252 trading days in a year
((1+np.mean(StockPrices["Returns"]))**252)-1
```

*Calculating Standard deviation and Variance in Python:*

- Standard deviation, represented by the symbol $\sigma$ is often referred to as **volatility** in financial analysis.
- An investment with a higher $\sigma$ is viewed as a higher risk investment.
- Measure of the dispersion of returns.
- Variance $= \sigma^2$

```python
import numpy as np

# Standard deviation
np.std(StockPrices["Returns"])

# Variance
np.std(StockPrices["Returns"])**2
```

Scaling volatility:

**Volatility scales with the square root of time.**

You can normally assume 252 trading days in a given year and 21 trading days in a given month:

$\sigma_{annual} = \sigma_{daily}*\sqrt{252}$

$\sigma_{monthly} = \sigma_{daily}*\sqrt{21}$

```python
import nump as np

np.std(StockPrices["Returns"]) * np.sqrt(252)
```

### Skewness and Kurtosis

Skewness is the third moment of a distribution after mean and variance.

**Negative skew**: The mass of the distribution is concentrated on the right. Usually a right-leaning curve.

**Positive skew**: The mass of the distribution is concentrated on the left. Usually a left-leaning curve.

In finance, we would tend to want a positive skewness.

*Calculating Skewness in Python:*

```python
from scipy.stats import skew

skew(StockData["Returns"].dropna())
```

Kurtosis is a measure of the thickness of the tails of a distribution.

Most financial returns are **leptokurtic** $\implies$ positive excess kurtosis (kurtosis > 3).

$Excess\:Kurtosis = K - 3$

Common for kurtosis functions to evaluate excess kurtosis rather than kurtosis since it is common to want to compare with a normal distribution.

$Excess\:Kurtosis > 0 \implies$ Probability of outliers higher than in normal distribution

*Calculating Excess Kurtosis in Python:*

```python
from scipy.stats import kurtosis

kurtosis(StockData["Returns"].dropna())
```

In a finance context, high excess kurtosis is an indication of high risk implying large movements in returns happen often.

### Testing for Normality - Shapiro Wilk test

For distributions where the skewness and excess are close to 0 but not quite, we can use the **Shapiro-Wilk** test to evaluate the normality of the distribution.

The null hypothesis of the Shapiro-Wilk test is that the data are normally distributed.

```python
from scipy import stats

p_value = stats.shapiro(StockData["Returns"].dropna())[1]

if p_value <= 0.05:
  print("Null hypothesis of normality is rejected")
else:
  print("Failed to reject null hypothesis of normality at 5% significance level)
```

## Portfolio Investing

### Portfolio composition and backtesting

*Portfolio Return Formula*:

$R_p = R_{a_1}\omega_{a_1} + R_{a_2}\omega_{a_2} + ... + R_{a_n}\omega_{a_n}$

- $R_p$: Portfolio return
- $R_{a_n}$: Return for asset n (discrete return)
- $\omega_{a_n}$: Weight for asset n

*Calculating Portfolio Returns in Python:*

```python
import numpy as np

portfolio_weights = np.array([0.25, 0.35, 0.10, 0.20, 0.10])
port_ret = StockReturns.mul(portfolio_weights, axis=1).sum(axis=1)

StockReturns["Portfolio"] = port_ret
```

Any good strategy should outperform an equally weighted portfolio.

```python
import numpy as np

numstocks = 5
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)
StockReturns.iloc[:,0:numstocks].mul(portfolio_weights_ew, axis=1).sum(axis=1)
```

*Plotting Portfolio Returns:*

Daily Returns:
```python
StockPrices["Returns"] = StockPrices["Adj Close"].pct_change()
StockReturns = StockPrices["Returns"]
StockReturns.plot()
```

*Cumulative Returns:*
```python
import matplotlib.pyplot as plt

CumulativeReturns = ((1 + StockReturns).cumprod() - 1)
CumulativeReturns[["Portfolio", "Portfolio_EW"]].plot()
```

**Market Capitalisation Weighting**

One common method of weighting portfolios is by **market capitalisation**. For example, the S&P 500 index is modelled on a market capitalisation weighted portfolio of the top 500 US stocks.

**Market capitalisation**: The value of a company's publicly traded shares.

*Calculating Market Cap Weighting*:

Market cap weight of a given stock $n$,

$\omega_{mcap_n} = \frac{mcap_n}{\sum_{i=1}^{n}mcap_i}$

```python
import numpy as np

market_caps = np.array([100, 200, 100, 100])
mcap_weights = market_caps/sum(market_caps)
```

### Correlation and co-variance

**Pearson correlation:**
| coefficient | correlation                  |
| ---         | ---                          |
| -1          | perfect negative correlation |
| 0           | no correlation               |
| 1           | perfect positive correlation |

We can plot a heatmap to explore the correlation between various stocks in our portfolio:

```python
import seaborn as sns

correlation_matrix = StockReturns.corr()

# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()
```

**Portfolio standard deviation:**

$\sigma_p = \sqrt{\omega_1^2\sigma_1^2 + \omega_2^2\sigma_2^2 + 2\omega_1\omega_2\rho_{1,2}\sigma_1\sigma_2}$

$\sigma_p$: Portfolio standard deviation

$\omega$: Asset weight

$\sigma$: Asset volatility

$\rho_{1,2}$: Correlation between assets 1 and 2

Works well with two variables, but gets messy quickly with larger numbers.
This is where the co-variance matrix comes in.

$\sigma_p = \sqrt{\omega_T \cdot \sum \cdot \omega}$

$\sigma_p$ : Portfolio volatility (standard deviation)

$\sum$ : Co-variance matrix of returns

$\omega$ : Portfolio weights

$\omega_T$ : Transposed portfolio weights

$\cdot$ : Dot product

```python
import numpy as np

#Calculating portfolio volatility
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
```

**Co-variance Matrix:**

Co-variance mesaures the joint variability of two random variables. 
Used in finance for portfolio optimisation and risk management.

*Calculating the Co-variance Matrix in Python:*

```python
cov_mat = StockReturns.cov()

# Annualised covariance matrix
cov_mat_annual = cov_mat * 252
```
### Markowitz Portfolios

**Sharpe Ratio**: A measure of risk-adjusted return.

$S = \frac{R_a - r_f}{\sigma_a}$

$S$ : Sharpe Ratio

$R_a$ : Asset return

$r_f$ : Risk-free rate of return

$\sigma_a$ : Asset volatility

Measurement of return for each incremental unit of risk. Allows comparison of different portfolios.

**Efficient Frontier**: Highest expected return for a given level of risk. Any portfolio in the efficient frontier is an optimum portfolio.

![Efficient Frontier](https://upload.wikimedia.org/wikipedia/commons/e/e1/Markowitz_frontier.jpg "Efficient Frontier")

The **tangency portfolio** is on the tangent to the efficient frontier with the highest Sharpe ratio. Also known as **MSR**, *Max Sharpe Ratio* . This tangent is known as the **Capital allocation line**, who's y-intercept marks the risk-free rate.

The **GMV**, *Global Minimum Volatility* portfolio is the portfolio on the far left edge of the plot (tagent parallel to y axis). This is the portfolio with the lowest volatility.

**Choosing a Portfolio**:
- Best to pick a portfolio on the bounding edge of the efficient frontier.
- Higher return is available at the cost of higher risk.

*Selecting the MSR in Python:*

```python
# Risk free rate
risk_free = 0

# Calculate the Sharpe Ratio for each asset
RandomPortfolios['Sharpe'] = (RandomPortfolios["Returns"] - risk_free) / RandomPortfolios["Volatility"]

# Print the range of Sharpe ratios
print(RandomPortfolios['Sharpe'].describe()[['min', 'max']])

# Sort the portfolios by Sharpe ratio
sorted_portfolios = RandomPortfolios.sort_values(by=['Sharpe'], ascending=False)

# Extract the corresponding weights
MSR_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the MSR weights as a numpy array
MSR_weights_array = np.array(MSR_weights)

# Calculate the MSR portfolio returns
StockReturns['Portfolio_MSR'] = StockReturns.iloc[:, 0:numstocks].mul(MSR_weights_array, axis=1).sum(axis=1)

# Plot the cumulative returns
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_MSR'])
```

The **MSR** is often quite erratic. Just because a portfolio had a good historic Sharpe ratio, it doesn't guarantee that the portfolio will have a good Sharpe Ratio moving forward.

*Selecting the GMV in Python:*

```python
# Sort the portfolios by volatility
sorted_portfolios = RandomPortfolios.sort_values(by=['Volatility'], ascending=True)

# Extract the corresponding weights
GMV_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the GMV weights as a numpy array
GMV_weights_array = np.array(GMV_weights)

# Calculate the GMV portfolio returns
StockReturns['Portfolio_GMV'] = StockReturns.iloc[:, 0:numstocks].mul(GMV_weights_array, axis=1).sum(axis=1)

# Plot the cumulative returns
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_MSR', 'Portfolio_GMV'])
```

Returns are very hard to predict, but volatilities and correlations tend to be more stable over time. This means that the GMV portfolio often outperforms the MSR portfolios out of sample even though the MSR would outperform quite significantly in-sample. Of course, out of sample results are what really matters in finance.

## Factor Investing

Factor analysis is the practice of using known factors such as returns of large stocks or growth stocks as independent variables in your analysis of portfolio returns. There are many different factor models which may be used.

### Capital Asset Pricing Model (CAPM)

The Capital Asset Pricing Model is the fundamental building block for many other asset pricing models and factor models in finance.

**Excess Returns:**

$Excess Returns = Returns - Risk Free Return$

E.g

*Investing in Brazil:*

- Return on deposits from bank = 15%
- Assuming 10% return on stock market

> $Excess Return_{Br} = 10\% - 15\% = -5\%$

*Investing in US:*
- Return on deposits from bank = 3%
- Assuming 10% return on stock market

> $Excess Return_{US} = 10\% - 3\% = 7\%$

(Since 2008 recession US Excess Return has been close to zero, but rising slowly and steadily since)

**Capital Asset Pricing Model:**

$E(R_p) - RF = \beta_P(E(R_M) - RF)$

- $E(R_p) - RF$: Excess expected return of a stock portfolio P
- $E(R_M) - RF$: Excess expected return of the broad market portfolio B
- $RF$: Regional risk-free rate
- $\beta_P$: Portfolio beta, or exposure, to the broad market portfolio B

*Calculating Beta using co-variance:*

$\beta_P=\frac{Cov(R_P, R_B)}{Var(R_B)}$

- $\beta_P$: Portfolio beta
- $Cov(R_P, R_B)$: Co-variance between the portfolio (P) and the benchmark market index (B).
- $Var(R_B)$: Variance of the benchmark market index

*Calculating Beta using co-variance in Python:*

```python
covariance_matrix = Data[["Port_Excess", "Mkt_Excess"]].cov()
covariance_coefficient = covariance_matrix.iloc[0,1]
benchmark_variance = Data["Mkt_Excess"].var()
portfolio_beta = covariance_coefficient / benchmark_variance
```

*Calculating Beta using linear regression:*

Beta corresponds to the coefficient of the linear regression between portfolio and market excess returns.

```python
import statsmodels.formula.api as smf

model = smf.ols(formula='Port_Excess ~ Mkt_Excess', data=Data)
fit = model.fit()

beta = fit.params["Mkt_Excess"]
r_squared = fit.rsquared
adjusted_r_squared = fit.rsquared_adj # penalises for number of parameters to prevent overfitting
```

### Alpha and multi-factor models

**Fama-French 3 factor model:**

$R_P = RF + \beta(R_M - RF) + b_{SML} + b_{HML} \cdot HML + \alpha$

- $SMB$: Small minus big factor (small stocks tend to outperform big stocks, SMB represents small size premium)
- $b_{SMB}$: Exposure to the SMB factor
- $HML$ High minus low factor (value vs growth- cyclical in nature since during the crisis value stocks outperformed, but during bull runs growth stocks outperform)
- $b_{HML}$: Exposure to HMB factor
- $\alpha$: Performance which is unexplained by other factors
- $\beta_M$: Beta to broad market portfolio B

*Calculating Fama-French Model in Python:*

```python
import statsmodels.formula.api as smf

model = smf.ols(formula='Port_Excess ~ Mkt_Excess + SMB + HML',
                data=Data)

fit = model.fit()
adjusted_r_squared = fit.rsquared_adj
```

Fama-French outpreforms CAPM model, explaining 90% of portfolio variance vs 70% for CAPM on average.

This makes sense since investors should be theoretically rewarded for exposure to small cap stocks instead of large caps and distressed/value stocks rather than popular and premium growth stocks. More risk, more reward.

We can evaluate the statistical significance of the `HML` and `SMB` coefficients by extracting the p-values:

```python
hml_sig = (fit.pvalues["HML"] < 0.05)

smb_sig = (fit.pvalues["SMB"] < 0.05)
```

The coefficients themselves may be extracted by:

```python
hml = fit.params["HML"]

smb = fit.params["SMB"]
```

*Alpha and the efficient market hypothesis:*

**Alpha** -  Anything which can't be explained by the beta, size of value factors in the model (essentially an error term). Interpreted as outperfomance due to skill, luck or timing.

For every fund with a postive alpha, there will be another with negative alpha, since the weighted sum of all alphas in a market must be zero (weighted sum of returns of all investors simply equals market portfolio).

Some economists believe the 'efficient market hypothesis' which states that the market prices all information and any perceived alpha is simply the result of a missing factor in a more complex economic pricing model.

```python
portfolio_alpha = fit.params["Intercept"]
portfolio_alpha_annualized = ((1 + portfolio_alpha) ** 252) - 1
```

### Expanding the 3-factor model

In 2015, Fama and French extended their model, adding two additional factors:

- $RMW$: Profitability
- $CMA$: Investment

The $RMW$ factor represents the returns of companies with high operating profitability versus those with low operating profitability.

The $CMA$ factor represents the returns of companies with aggresive investments versus those who are more conservative.

```python
import statsmodels.formula.api as smf

model = smf.ols(
          formula='Port_Excess ~ Mkt_Excess + SMB + HML + RMW + CMA',
          data=Data)

fit = model.fit()

adjusted_r_squared = fit.rsquared_adj
```

Adding additional factors is as simple as adding additional coefficients to the model. A large amount of quantitative investment research goes into hypothesing and testing additional factors to this model.

## Value at Risk

### Estimating Tail Risk

Tail risk is the risk of extreme investment outcomes, most notably on the negative side of a distribution.

**Historical Drawdown:**

**Drawdown** is the percentage loss from the highest cumulative historical point.

$Drawdown = \frac{r_t}{RM} - 1$

- $r_t$: Cumulative return at time t
- $RM$: Running maximum

Ideal investments would have little to no drawdown, growing consistently over time. But in reality we can expect some level of drawdown quite frequently in most portfolios.

*Calculating Historical Drawdown in Python:*

```python
running_max = np.maximum.accumulate(cum_rets)
running_max[running_max < 1] = 1
drawdown = (cum_rets) / running_max - 1
```

**Historical Value at Risk:**

**Value at risk**, or Var, is a threshold with a given confidence level that losses will not exceed a certain level.

Var is commonly quoted with quantiles/percentages such as 95, 99 and 99.9.

E.g. Var(95) = -2.3% $\implies$ 95% certain that losses won't exceed -2.3% on a given day, based on historical values.

*Calculating Var in Python:*

```python
var_level = 95

var_95 = np.percentile(StockReturns, 100 - var_level)
```

**Historical Expected Shortfall:**

**Conditional value at risk**, or CVar, is an estimate of expected losses sustained in the worst 1-x% of scenarios.

CVar is commonly quoted with quantiles/percentages such as 95, 99 and 99.9.

E.g. CVar(95) = -2.5% $\implies$ In the worst 5% of cases, losses were on average in excess of -2.5% historically.

Essentially the same as taking the average losses of cases exceeding the Var(95)

*Calculating CVar in Python:*

```python
var_level = 95

var_95 = np.percentile(StockReturns, 100 - var_level)
cvar_95 = StockReturns[StockReturns <= var_95].mean()
```

### Var Extensions

**Var quantiles**
Depending on the chosen quantile we will get varying forecasts of potential loses. Only using the 95th quantile would mean that we'd underestimate the risk in 5% of cases, equally using too extreme of a quantile could lead to an overestimate of risk, leading to loses due to over cautiousness. It is therefore common to test various quantile parameters to strike the right balance between risk and reward.

**Empirical Assumptions**
Up to this point we have only considered empirical historical values, i.e. values which have actually occurred. To simulate a value which has never occurred before, we must therefore assume a probability distribution and extract quantiles from this distribution.

*Parametric VaR in Python:*

```python
mu = np.mean(StockReturns)
std = np.std(StockReturns)
confidence_level = 0.05 # Calculating VaR(95)

VaR = norm.ppf(confidence_level, mu, std)
```

*Scaling Risk in Python:*

Estimates of VaR scale with the square root of time, due to their compounding nature.

```python
forecast_days = 5
forecast_var95_5day = var_95 * np.sqrt(forecast_days)
```

### Random Walks

Random, or stochastic, movement are apparent throughout nature.Tidal waves, earthquakes, crime, particle physics and the stock market, are all random in nature but may still be modelled and predicted using mathematics.

*Random Walks in Python:*

```python
mu = np.mean(StockReturns)
std = mp.mean(StockReturns)

T = 252 # Forecast period
S0 = 10 # Starting stock price

rand_rets = np.random.normal(mu, std, T) + 1
forecasted_values = S0 + (rand_rets.cumprod())
```

**Monte Carlo Simulations** - Repeating many random walk simulations we can compute a large number of possible outcomes, this is known as a Monte Carlo simulation.

We can then study the outcomes of Monte Carlo simulations in the same way we previously studied historical values, to compute value at risk etc. The only difference being that we can now create as many simulations as we want and tweak them to have the characteristics we desire.

*Monte Carlo Simulation in Python:*

```python
mu = 0.0005
vol = 0.001
T = 252

sim_returns = []

for i in range(100):
  rand_rets = np.random.normal(mu, vol, T)
  sim_returns.append(rand_rets)

var_95 = np.percentile(sim_returns, 5)
```

