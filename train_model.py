import pandas as pd

def get_unique_players(csv_path: str) -> list:
    df = pd.read_csv(csv_path)

    # Collect all player names from relevant columns
    players = set()

    for col in ['striker', 'bowler', 'player_dismissed']:
        if col in df.columns:
            players.update(df[col].dropna().unique())

    # Return sorted list
    return sorted(players)

# Example usage:
if __name__ == "__main__":
    path = "deliveries.csv"  # replace with your actual file
    players = get_unique_players(path)
    print(players)
