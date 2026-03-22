# Commerce Intelligence

**A commercial intelligence platform for UK wholesale giftware retail.**

Live dashboard: https://commerce-intelligence.streamlit.app  
Built on: UCI Online Retail II dataset (776,842 transactions, 2009-2011)

---

## What This Is

Most commercial decisions in wholesale retail are still made manually.
A pricing analyst pulls a lever in a spreadsheet. An inventory planner
places a purchase order based on last quarter's formula. Nobody sees
the whole system at once.

This project builds that system.

Commerce Intelligence combines price elasticity modelling, demand
forecasting, live competitor price scraping, and a commercial
intelligence dashboard into a single connected platform. Every
component produces outputs that feed the next. The result is a unified
view of pricing, inventory risk, and market positioning across 12
product categories.

---

## Modules

### Pricing Intelligence
Log-log OLS regression on 2,480 SKU-customer segment combinations
using Volume-Weighted Average Price (VWAP) aggregation. Wholesale and
retail demand curves calculated separately to avoid Simpson's Paradox
from mixing customer types. Outputs elasticity coefficients,
commercial priority scores, and estimated annual opportunity per SKU.

Key finding: Retail customers show inelastic demand on 135 SKUs,
representing £3,911 estimated annual opportunity from a 10% price
increase. High-revenue wholesale SKUs are price-elastic, consistent
with a B2B environment where bulk buyers negotiate on price.

### Demand Forecasting
One LightGBM model per SKU across the top 50 products by revenue.
Complete week grid constructed to handle zero-sales weeks that would
otherwise corrupt lag features. Tweedie regression objective selected
for its mathematical suitability to zero-inflated intermittent demand
data. Date-based walk-forward validation with September 2011 cutoff,
testing on Q4 peak period.

Results: 80% of models outperform naive baseline. Median improvement
17.1% over naive. Stock-out risk report generated with CRITICAL, HIGH,
MEDIUM, LOW classifications and suggested reorder quantities.

### Competitor Intelligence
Live web scraper pulling product names and prices from Cox & Cox
across 9 category URLs using BeautifulSoup. Price cap of £150 applied
to Storage category to remove non-comparable furniture items.
Wholesale-to-retail gap analysis across all 12 categories using a
2.5x implied retail markup assumption.

Result: All 12 categories show greater than 100% wholesale-to-retail
gap, ranging from 303% (Doormats) to 2,708% (Bags), confirming that
wholesale customers operate with substantial retail margin. This
provides the basis for measured wholesale price increases.

### Commercial Overview
Unified dashboard combining elasticity, forecasting, and competitor
data in a single category-level view. Methodology and assumptions
documented transparently within the application.

---

## Architecture
```
data/raw/                    UCI Online Retail II dataset
    |
src/cleaning.py              776,842 clean rows, documented pipeline
    |
src/elasticity.py            VWAP, segmented OLS, vectorized
src/forecasting.py           LightGBM Tweedie, gap-filled weeks
src/scraper.py               Cox & Cox live scrape, gap analysis
    |
outputs/                     Pre-computed CSVs
    |
dashboard/                   Streamlit multi-page application
    |
docker/                      Containerised deployment
```

---

## Key Technical Decisions

**Why VWAP instead of mean price for elasticity:**
Simple mean price ignores transaction volume. A week with 1 unit at
£10 and 100 units at £5 produces a mean of £7.50 but a VWAP of £5.05.
Using mean price would corrupt the elasticity regression.

**Why Tweedie objective for forecasting:**
Default LightGBM regression (MSE) assumes normally distributed
targets. Retail weekly demand is zero-inflated with a long tail of
bulk orders. Tweedie distribution is mathematically designed for
exactly this data shape. Switching from MSE to Tweedie improved
median baseline improvement from 9.7% to 17.1%.

**Why separate elasticity by customer segment:**
The dataset mixes retail customers buying 1-5 units and wholesale
buyers ordering 50-5,000 units. Aggregating both produces extreme
coefficients because the model mistakes different customer tiers for
price sensitivity. Segmentation produces separate, more credible
demand curves for each customer type.

**Why date-based rather than row-based train-test split:**
Row-based 80/20 splits produce different time periods for different
SKUs, making test metrics incomparable. A hard date split at September
2011 gives every SKU the same test window, including the Q4 peak
which is the commercially critical period to forecast correctly.

**Known limitations:**
Price elasticity estimates are directional rather than precise. The
dataset lacks genuine price change events, so most price variation
reflects different customer tier pricing on the same day rather than
deliberate pricing decisions. A production system would require
dedicated price-change event logs for unbiased estimation. Demand
models are trained on a maximum of 88 weeks of data per SKU. More
history would improve accuracy.

---

## Running Locally
```bash
git clone https://github.com/csharma01/Commerce-Intelligence.git
cd Commerce-Intelligence
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/Home.py
```

---

## Running with Docker
```bash
docker build -f docker/Dockerfile -t commerce-intelligence .
docker-compose -f docker/docker-compose.yml up
```

Open http://localhost:8501

---

## Data

**Transactions:** UCI Online Retail II dataset. Real UK wholesale
giftware retailer, December 2009 to November 2011, 1,067,371 raw rows.
Available at: https://archive.ics.uci.edu/dataset/502/online+retail+ii

**Competitor prices:** Publicly available retail listings from
coxandcox.co.uk scraped via HTTP requests. Used solely for
non-commercial portfolio analysis.

---

## What a Production Version Would Add

- Dedicated price-change event logging for unbiased elasticity
- Daily automated competitor price refresh via scheduled scraper
- Retail customer elasticity using monthly aggregation to address
  data sparsity at weekly SKU level
- Separate wholesale and retail demand forecasting models
- Automated reorder triggers connected to purchasing systems
- Multi-market elasticity comparison as the business expands

---

## Tests
```bash
$env:PYTHONPATH = "."; pytest tests/ -v
```

Nine tests across cleaning, elasticity, and forecasting modules.

---

## Stack

Python, Pandas, NumPy, LightGBM, statsmodels, scikit-learn,
Streamlit, Plotly, BeautifulSoup, Docker, Git