import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load and clean the dataset
file_path = 'historical_data_2010-2025.csv'
data = pd.read_csv(file_path)

# Drop rows where home or away team or score information is missing
cleaned_data = data.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])

# Create a new column for the match outcome (1 if home team wins, 0 if away team wins)
cleaned_data['home_win'] = (cleaned_data['home_score'] > cleaned_data['away_score']).astype(int)

# Function to calculate the win percentage for a team (overall, not just the last 5 matches)
def calculate_win_percentage(team_name, data, is_home=True):
    if is_home:
        team_data = data[data['home_team'] == team_name]
        win_count = team_data['home_win'].sum()
        total_games = len(team_data)
    else:
        team_data = data[data['away_team'] == team_name]
        win_count = team_data['home_win'].apply(lambda x: 1 - x).sum()  # Away win is inverse of home win
        total_games = len(team_data)
    
    return win_count / total_games if total_games > 0 else 0

# Create overall win percentages for both home and away teams
cleaned_data['home_team_win_pct'] = cleaned_data['home_team'].apply(lambda x: calculate_win_percentage(x, cleaned_data, is_home=True))
cleaned_data['away_team_win_pct'] = cleaned_data['away_team'].apply(lambda x: calculate_win_percentage(x, cleaned_data, is_home=False))

# Convert the 'date' column to datetime format for sorting purposes
cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')

# Function to calculate the win percentage for the last 5 matches of a team
def calculate_last_5_matches_win_pct(team_name, data, is_home=True):
    # Filter data based on home or away team
    if is_home:
        team_data = data[data['home_team'] == team_name]
    else:
        team_data = data[data['away_team'] == team_name]
    
    # Sort the matches by date to get the latest 5
    team_data = team_data.sort_values('date', ascending=False)
    
    # Select the last 5 matches
    last_5_matches = team_data.head(5)
    
    # Calculate the win count in the last 5 matches
    if is_home:
        win_count = last_5_matches['home_win'].sum()
    else:
        win_count = last_5_matches['home_win'].apply(lambda x: 1 - x).sum()  # Away win is inverse of home win
    
    return win_count / 5 if len(last_5_matches) == 5 else 0  # Return win percentage

# Feature engineering: Add home team and away team performance metrics (including last 5 matches)
cleaned_data['home_team_last_5_win_pct'] = cleaned_data['home_team'].apply(lambda x: calculate_last_5_matches_win_pct(x, cleaned_data, is_home=True))
cleaned_data['away_team_last_5_win_pct'] = cleaned_data['away_team'].apply(lambda x: calculate_last_5_matches_win_pct(x, cleaned_data, is_home=False))

# Prepare features and target variable
X = cleaned_data[['home_team_win_pct', 'away_team_win_pct', 'home_team_last_5_win_pct', 'away_team_last_5_win_pct']]
y = cleaned_data['home_win']  # Target variable (home team win or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Function to predict match winner
def predict_match_winner(home_team, away_team, model, data):
    # Extract the win percentages for both the home and away teams
    home_team_win_pct = calculate_win_percentage(home_team, data, is_home=True)
    away_team_win_pct = calculate_win_percentage(away_team, data, is_home=False)
    
    # Calculate last 5 matches win percentages
    home_team_last_5_win_pct = calculate_last_5_matches_win_pct(home_team, data, is_home=True)
    away_team_last_5_win_pct = calculate_last_5_matches_win_pct(away_team, data, is_home=False)

    # Prepare the features for prediction (match the same features used for training)
    features = [[home_team_win_pct, away_team_win_pct, home_team_last_5_win_pct, away_team_last_5_win_pct]]
    # Predict the probability of the home team winning
    proba = model.predict_proba(features)
    
    # Return the probability of each team winning
    home_team_prob = proba[0][1]  # Probability that home team wins
    away_team_prob = 1 - home_team_prob  # Probability that away team wins
    
    return home_team_prob, away_team_prob

# Example usage: Predict the outcome for two teams
home_team_prob, away_team_prob = predict_match_winner("Cincinnati Bengals", "Dallas Cowboys", model, cleaned_data)
print(f"Home Team Probability: {home_team_prob:.4f}")
print(f"Away Team Probability: {away_team_prob:.4f}")

