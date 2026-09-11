# AI Stock Dashboard - Setup Instructions

## Prerequisites
- Python 3.x installed on your system

## Setup Steps

### 1. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 2. Install Required Dependencies

Once the virtual environment is activated, install all required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure LLM API Keys (for news sentiment)

[`src/sentiment_engine.py`](src/sentiment_engine.py) scores news sentiment with an LLM.
Create a `.env` file at the project root (it is git-ignored) with **one** of:

```bash
# Anthropic Claude
ANTHROPIC_API_KEY=your_key_here

# ...or OpenAI
OPENAI_API_KEY=your_key_here
```

Optional overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SENTIMENT_LLM_PROVIDER` | auto-detect (Claude first) | Force `claude` or `openai` when both keys are set |
| `CLAUDE_MODEL` | `claude-opus-5` | Claude model id |
| `CLAUDE_EFFORT` | `medium` | Claude reasoning depth: `low`, `medium`, `high`, `xhigh` or `max` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model id |

Without a key the engine still runs: news fetching needs no credentials, and
scoring falls back to a neutral `0.0` with `status="no_api_key"`.

### 4. Verify Installation

You can verify that all packages are installed correctly by running:

```bash
pip list
```

You should see the following packages:
- pandas
- yfinance
- numpy
- python-dotenv

### 5. Deactivate Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
ai-stock-dashboard/
├── .venv/                      # Python virtual environment
├── data/                       # Directory for raw CSV/database files
├── models/                     # Trained model artefacts (joblib)
├── src/                        # Source code directory
│   ├── data_loader.py          # Fetch, clean and persist OHLCV data
│   ├── feature_engineering.py  # Technical indicators
│   ├── ml_engine.py            # Model training + ml_score
│   ├── sentiment_engine.py     # News fetching + LLM sentiment_score
│   └── signal_blender.py       # Score blending + signal mapping
├── ui/                         # Streamlit presentation layer
│   ├── config.py               # Ticker universe, date ranges, validation
│   ├── sidebar.py              # Selection controls -> Selection
│   ├── layout.py               # Main-area section containers
│   ├── market_table.py         # Ranked overview table + row selection
│   ├── signal_card.py          # The Daily Signal card
│   ├── price_chart.py          # Candlestick + volume chart
│   └── breakdown.py            # ML / sentiment component columns
├── app.py                      # Streamlit entry point
├── market_scan.py              # Multi-ticker scan behind the overview table
├── run_analysis.py             # End-to-end analysis driver (CLI + library)
├── .env                        # API keys (git-ignored, create yourself)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── SETUP.md                    # This setup guide
```

## Running the Dashboard

With the virtual environment activated:

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

The page leads with a **Market Overview** table: the full preset universe scored
end to end and ranked by conviction, so Strong Buy and Strong Sell sit at the
top and weak or neutral names at the bottom. It shows the top 20 by default,
with a *Show all* toggle underneath. Click any column header to re-sort.

Below the table, one stock is unpacked in detail — signal card, price chart and
the two component scores. Click a table row to point the detail sections at that
stock, or pick a ticker from the sidebar dropdown, or type any yfinance symbol
into the custom box (Indian listings need the `.NS` suffix), then choose how far
back to pull daily bars.

**The first load is the slow one.** Every row runs a price fetch, a model
inference and an LLM sentiment call, so a cold scan of the universe takes about
a minute behind its progress bar. Rows are then cached per symbol for 30
minutes, and the ticker you click is already scored — clicking a row costs no
fetch and no LLM call. To skip the LLM leg entirely, leave the API keys unset:
sentiment falls back to a neutral `0.0` and the scan finishes in seconds.

For a headless run without opening a browser:

```bash
streamlit run app.py --server.headless true
```

## Running a One-Off Analysis

The pipeline is also usable straight from the terminal, independent of the UI:

```bash
python run_analysis.py --ticker AAPL
python run_analysis.py --ticker RELIANCE.NS --lookback-days 1095
```

The overview table's scan is scriptable the same way, and prints the same
ranking the dashboard draws:

```bash
python market_scan.py AAPL MSFT RELIANCE.NS
python market_scan.py AAPL NVDA --news-limit 5 --workers 4
```
