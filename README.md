# Crypto Correlation Analyzer

## Research question
How do correlations among major cryptocurrencies evolve through market regimes, and can these correlation dynamics be used to detect periods of market stress?

## Setup and Usage

To ensure reproducibility, please use a virtual environment.

### 1. Clone the repository
git clone https://github.com/Max8211/crypto-correlation-analyser.git
cd crypto-correlation-analyser

### 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies
Run the following command to install the required software environment:

pip install -r requirements.txt

### 4. Execute Analysis
Run this command to regenerate all data processing, model training, and results:

python main.py

Expected output: Correlation Analysis, containing:

1. Bitcoin Correlation Summary
2. Overall Crypto-Market Correlation Summary
3. Market Regime Analysis
4. Clustering & PCA
5. Supervised regime classification

## Project Structure
crypto-correlation-analyser/
├── main.py                     # Main entry point (runs full pipeline)
├── requirements.txt            # Python dependencies
├── PROPOSAL.md                 # Initial project proposal
├── AI_USAGE.md                 # Documentation of AI tools used          
├── src/                        # Source code modules
│   ├── merge_data.py           # Merges raw CSVs into a clean dataset
│   ├── returns.py              # Computes daily percentage returns
│   ├── eda.py                  # Exploratory Data Analysis 
│   ├── correlation_analysis.py # Rolling & EWMA correlation 
│   ├── regime_analysis.py      # Detects Normal vs. Stress market regimes
│   ├── pca.py                  # Principal Component Analysis
│   ├── clustering.py           # K-Means clustering 
│   └── supervised_regime_classification.py # Random Forest model
├── data/
│   └── raw/                    # Original 10 raw datasets
└── results/
    ├── data/                   # Base processed datasets 
    ├── outputs/                # Computed metrics 
    └── figures/                # Generated plots 

## Requirements
- Python 3.10+
- scikit-learn, pandas, matplotlib, seaborn, numpy, scipy

