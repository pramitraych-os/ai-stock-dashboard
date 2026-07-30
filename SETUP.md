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
# Google Gemini
GEMINI_API_KEY=your_key_here

# ...or OpenAI
OPENAI_API_KEY=your_key_here
```

Optional overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SENTIMENT_LLM_PROVIDER` | auto-detect (Gemini first) | Force `gemini` or `openai` when both keys are set |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model id |
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
│   └── sentiment_engine.py     # News fetching + LLM sentiment_score
├── .env                        # API keys (git-ignored, create yourself)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── SETUP.md                    # This setup guide
```

## Next Steps

- Add your data files to the [`data/`](data/) directory
- Implement data loading logic in [`src/data_loader.py`](src/data_loader.py)
- Create additional scripts in the [`src/`](src/) directory as needed
