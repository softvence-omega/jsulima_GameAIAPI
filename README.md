# GameAI API

A FastAPI-based sports analytics and prediction API that provides machine learning-powered insights for MLB (Major League Baseball) and NFL (National Football League) games.

## Features

### MLB (Major League Baseball)
- **Game Predictions**: Predict outcomes of upcoming MLB games
- **Lineup Predictions**: Optimize team lineups for better performance
- **Head-to-Head Analysis**: Compare teams and predict matchup outcomes
- **Top Performer Analysis**: Identify best batters and pitchers
- **Win Percentage Calculations**: Calculate team win probabilities
- **Batter/Pitcher Performance**: Predict individual player statistics

### NFL (National Football League)
- **Win Predictions**: Predict NFL game outcomes
- **Lineup Optimization**: Suggest optimal player lineups
- **Top Performer Predictions**: Identify key players for upcoming games
- **Head-to-Head Comparisons**: Analyze team matchups
- **Win Percentage Analysis**: Calculate team win probabilities
- **Upcoming Games**: Get schedule information for upcoming NFL games

## Technology Stack

- **Framework**: FastAPI
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **HTTP Client**: httpx, requests
- **Scheduling**: APScheduler
- **Web Server**: uvicorn, gunicorn
- **Data Visualization**: matplotlib, seaborn
- **Environment Management**: python-dotenv

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Roksana18cse04/GameAIAPI.git
   cd GameAIAPI
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

5. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GOALSERVE_API_KEY=your_goalserve_api_key_here
   GOALSERVE_BASE_URL=https://www.goalserve.com/getfeed/
   ```

## Running the Application

### Development Mode
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **Interactive API Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

## API Endpoints

### MLB Endpoints
- `GET /` - Welcome message and API information
- `POST /predict/mlb/games` - Predict MLB game outcomes
- `POST /predict/mlb/lineup` - Get optimal MLB lineup predictions
- `POST /predict/mlb/head-to-head` - MLB head-to-head team analysis
- `POST /predict/mlb/top_batter_pitcher` - Get top batter and pitcher predictions
- `GET /predict/mlb/win-percentages` - Calculate MLB team win percentages

### NFL Endpoints
- `POST /predict/nfl/win-prediction` - Predict NFL game outcomes
- `POST /predict/nfl/lineup` - Get optimal NFL lineup predictions
- `POST /predict/nfl/top-performers` - Predict top NFL performers
- `POST /predict/nfl/head-to-head` - NFL head-to-head team analysis
- `GET /predict/nfl/win-percentages` - Calculate NFL team win percentages
- `GET /predict/nfl/upcoming-games` - Get upcoming NFL games

## Usage Examples

### Get MLB Win Percentages
```bash
curl -X GET "http://localhost:8000/predict/mlb/win-percentages"
```

### Predict NFL Game Outcome
```bash
curl -X POST "http://localhost:8000/predict/nfl/win-prediction" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Kansas City Chiefs",
    "away_team": "Buffalo Bills"
  }'
```

### Get MLB Top Performers
```bash
curl -X POST "http://localhost:8000/predict/mlb/top_batter_pitcher" \
  -H "Content-Type: application/json" \
  -d '{
    "hometeam": "New York Yankees",
    "awayteam": "Boston Red Sox"
  }'
```

## Project Structure

```
GameAIAPI/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/          # API route handlers
│   ├── data/                       # Data storage
│   │   ├── MLB/                    # MLB datasets and notebooks
│   │   └── NFL/                    # NFL datasets and notebooks
│   ├── services/                   # Business logic and ML models
│   │   ├── MLB/                    # MLB-specific services
│   │   └── NFL/                    # NFL-specific services
│   ├── schemas/                    # Pydantic data models
│   ├── tests/                      # Unit tests
│   ├── utils/                      # Utility functions
│   ├── config.py                   # Configuration settings
│   ├── main.py                     # FastAPI application entry point
│   └── scheduler.py                # Background task scheduler
├── models/                         # Trained ML models
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup configuration
└── README.md                      # Project documentation
```

## Machine Learning Models

The API uses various machine learning algorithms for predictions:

- **XGBoost**: For win percentage calculations and game outcome predictions
- **Random Forest**: For player performance predictions
- **Linear Regression**: For statistical analysis and projections
- **Ensemble Methods**: Combining multiple models for improved accuracy

## Data Sources

- **GoalServe API**: Live scores, player stats, team information
- **Automated Data Collection**: Scheduled data fetching and model retraining
- **Historical Data**: Historical game data for model training

## Configuration

The application uses environment variables for configuration:

- `GOALSERVE_API_KEY`: API key for GoalServe sports data
- `GOALSERVE_BASE_URL`: Base URL for GoalServe API
- `PORT`: Server port (defaults to 8000)

## Testing

Run tests using:
```bash
python -m pytest app/tests/
```

## Development

### Adding New Features

1. Create new endpoint in `app/api/v1/endpoints/`
2. Add business logic in `app/services/`
3. Define data schemas in `app/schemas/`
4. Update routing in `app/main.py`

### Model Training

Models are automatically retrained on a schedule. Manual training can be triggered through the respective service modules in `app/services/`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is developed by Softvence Omega.

## Support

For support and questions, please refer to the API documentation at `/docs` when running the application.

---

**Welcome to GameAI API!** 🏈⚾ Use the `/docs` endpoint to explore all available endpoints and start making predictions!
