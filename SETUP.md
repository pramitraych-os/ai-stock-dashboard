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

### 3. Verify Installation

You can verify that all packages are installed correctly by running:

```bash
pip list
```

You should see the following packages:
- pandas
- yfinance
- numpy
- python-dotenv

### 4. Deactivate Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
ai-stock-dashboard/
├── .venv/              # Python virtual environment
├── data/               # Directory for raw CSV/database files
├── src/                # Source code directory
│   └── data_loader.py  # Data loading script
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── SETUP.md           # This setup guide
```

## Next Steps

- Add your data files to the [`data/`](data/) directory
- Implement data loading logic in [`src/data_loader.py`](src/data_loader.py)
- Create additional scripts in the [`src/`](src/) directory as needed
