# AI Stock Analysis Dashboard

A Streamlit dashboard that scores stocks with two independent signals and
blends them into one verdict:

- **Technical** -- a Random Forest trained on price/volume indicators (RSI,
  moving averages, MACD, Bollinger Bands), producing an `ml_score` in
  `[-1, +1]`.
- **Sentiment** -- recent news headlines scored by Anthropic Claude (or
  OpenAI as a fallback provider), producing an `ai_score` on the same scale.

The two are blended into a `blended_score` and mapped onto a six-way signal
ladder from **Strong Sell** to **Strong Buy**. If either leg fails or is
unavailable, the dashboard falls back to whichever leg *did* succeed rather
than diluting the verdict with a fake neutral reading, and says so with a
warning badge.

The **Market Overview** table ranks a preset universe of large-cap US and
NSE-listed Indian stocks by conviction; the detail section below it unpacks
one stock at a time -- signal card, price chart, and both component scores
with their supporting evidence (indicator readings, headlines).

## Quickstart (local)

```bash
git clone https://github.com/pramitraych-os/ai-stock-dashboard.git
cd ai-stock-dashboard

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then fill in ANTHROPIC_API_KEY, see below

streamlit run app.py
```

The app opens at <http://localhost:8501>. See [SETUP.md](SETUP.md) for the
full walkthrough (project structure, running a one-off CLI analysis, optional
environment variables, etc.) -- this file covers the essentials plus
deploying to Streamlit Community Cloud.

## Configuring `ANTHROPIC_API_KEY`

All configuration is resolved centrally in [`config.py`](config.py), so every
module reads secrets the same way regardless of where the app is running:

1. **Real environment variable / `.env` file** (local development).
2. **`st.secrets`** (Streamlit Community Cloud, or a local
   `.streamlit/secrets.toml`).

Neither is required to run the app -- without a key, sentiment scoring falls
back to a neutral `0.0` and the dashboard runs on the technical signal alone
-- but news sentiment needs one.

**Locally:** copy the template and fill in a real key:

```bash
cp .env.example .env
```

```dotenv
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Get a key from the [Anthropic Console](https://console.anthropic.com/settings/keys).
`.env` is git-ignored, so the real key never gets committed -- only
`.env.example`'s placeholder does. See `.env.example` for the full list of
optional variables (alternate provider, model overrides, yfinance network
settings).

**On Streamlit Community Cloud:** there is no `.env` file in a deployed app,
so set the same variable name in the app's **Secrets** instead -- see
[Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
below.

## Deploying to Streamlit Community Cloud

1. **Push this repository to GitHub** (public or private -- Streamlit
   Community Cloud can be authorized to access private repos).

   ```bash
   git push origin main
   ```

   Make sure `requirements.txt` is committed and up to date, and that `.env`
   is **not** tracked (`git status` should not show it -- `.gitignore`
   already excludes it).

2. **Sign in to [share.streamlit.io](https://share.streamlit.io)** with your
   GitHub account and authorize Streamlit to access the repository.

3. **Click "New app"** and fill in:
   - **Repository:** `pramitraych-os/ai-stock-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py`

   Under **Advanced settings**, optionally pin the Python version (this
   project is built against Python 3.12 -- see `.python-version`).

4. **Add the API key before the first deploy.** Still in Advanced settings
   (or afterwards via **Settings -> Secrets** on the deployed app), open the
   **Secrets** editor and add:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   ```

   This is plain TOML, one `KEY = "value"` pair per line -- the same optional
   variables documented in `.env.example` (`OPENAI_API_KEY`,
   `SENTIMENT_LLM_PROVIDER`, `CLAUDE_MODEL`, etc.) can go here too, in the
   same format. `config.py` reads this via `st.secrets` automatically; no
   code change is needed to switch between a local `.env` and Cloud secrets.

5. **Click "Deploy".** The first build installs `requirements.txt` from
   scratch, which takes a few minutes; the app then boots on `app.py`.

6. **Verify it works:** open the deployed URL, confirm the Market Overview
   table populates, and open one stock's detail section to confirm the
   Sentiment Score is a real number rather than "n/a" with a
   "no_api_key"/`auth_error` warning -- that would mean the secret didn't
   save or is misnamed.

7. **Updating the key later:** go to the app in
   [share.streamlit.io](https://share.streamlit.io) -> **⋮ (menu)** ->
   **Settings** -> **Secrets**, edit the TOML, and **Save** -- the app
   restarts automatically and picks up the new value on the next run.

8. **Redeploys:** every push to the tracked branch triggers an automatic
   redeploy. Secrets persist across redeploys -- they live in Streamlit's
   own store, not in the repository.

### Notes specific to this app

- The Random Forest models under `models/` are git-ignored (see
  `SETUP.md`'s project structure). On Streamlit Cloud, the first request for
  a ticker with no saved model trains one on the fly (`allow_training=True`
  by default) and keeps it in the container's cache for the life of that
  session -- it is not persisted back to the repo.
- Sentiment and price-fetch results are cached (`st.cache_data`, one hour;
  see `app.py`'s module docstring), so a cold Cloud instance's first load is
  the slow one. The sidebar's **Refresh Data** button clears the cache on
  demand.
