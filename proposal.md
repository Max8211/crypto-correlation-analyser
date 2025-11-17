# Crypto Market Correlation Analyzer: Exploring Interdependence and Market Structure

Category: Data Analysis & Visualization

Cryptocurrencies are fascinating assets. They go through phases of fast growth and sudden drops, often moving together in unpredictable ways. This volatility raises an important question: does the crypto market act as a single, connected system, or are there distinct coins/groups of coins that behave differently? Understanding how these relationships change over time can help investors and analysts better measure diversification and market risk. The question is : How are correlations between major cryptocurrencies structured and how do they evolve over time?

The goal of this project is to explore how correlations between major cryptocurrencies evolve and whether clusters of assets exist. I will collect daily closing prices of the main cryptocurrencies (Bitcoin, Ethereum, Binance Coin, Solana…) from 2018 to 2025 using public APIs such as CoinGecko. Using this data, I will compute returns and build time-varying correlation matrices through both rolling and exponentially weighted (EWMA) methods to capture how relationships shift over time.
These correlation features will be used in unsupervised machine learning models to detect structure in the market. Specifically, I will apply and compare clustering algorithms such as k-means, hierarchical clustering, and spectral clustering to identify groups of cryptocurrencies that tend to move together. Principal Component Analysis (PCA) will help measure how much of the overall market movement is explained by a few dominant factors. I will compare the clustering results using metrics like the silhouette score to evaluate which method best captures meaningful patterns.

The project will use Python libraries including pandas, NumPy, scikit-learn, and seaborn for data processing, modeling, and visualization. Expected outputs include heatmaps, cluster diagrams, and timelines showing how market relationships evolve.

The main challenges will involve data cleaning, handling missing values, and interpreting results in financial terms. The project will be successful if it clearly identifies how interdependence within the crypto market changes over time and provides insights into whether cryptocurrencies behave as one market or form separate groups.

