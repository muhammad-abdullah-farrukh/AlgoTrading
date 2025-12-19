# Trading Application Backend

Backend skeleton for the trading application built with FastAPI.

## Requirements

- Python 3.10+

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

3. Run the application:
```bash
python run.py
```

The server will start on `http://localhost:8000` by default.

## Endpoints

- `GET /health` - Health check endpoint returning `{"status": "ok"}`

## Database

The application uses SQLite for persistent storage. The database file (`trading.db`) is automatically created in the backend directory on first startup.

### Tables

- **ohlcv**: Stores OHLCV (Open, High, Low, Close, Volume) market data
- **trades**: Stores trade execution records
- **positions**: Stores trading positions
- **dataset_metadata**: Stores metadata about datasets

The database is automatically initialized on application startup. All data persists across restarts.

## Project Structure

```
backend/
├── run.py                 # Single entry point
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration management
│   ├── database.py       # Database connection & session management
│   ├── models.py         # SQLAlchemy database models
│   ├── dependencies.py   # FastAPI dependencies
│   ├── routers/
│   │   └── health.py     # Health check router
│   └── utils/
│       └── logging.py    # Logging utilities
├── trading.db            # SQLite database (created automatically)
└── requirements.txt      # Python dependencies
```
