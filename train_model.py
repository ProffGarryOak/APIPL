# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("matches.csv")

# Filter valid matches
df = df[df['match_winner'].notna() & (df['team1'] != df['team2'])]

# One-hot encode team names
teams = list(set(df['team1']).union(set(df['team2'])))
for team in teams:
    df[f'team1_{team}'] = (df['team1'] == team).astype(int)
    df[f'team2_{team}'] = (df['team2'] == team).astype(int)

X = df[[col for col in df.columns if col.startswith('team1_') or col.startswith('team2_')]]
y = df['match_winner']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open("winner_model.pkl", "wb") as f:
    pickle.dump((model, teams), f)
