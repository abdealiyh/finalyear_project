import pandas as pd
import streamlit as st
import plotly.express as px

# Read in the dataset
df = pd.read_csv('FINAL_MERGED_FINAL_DATASET.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add a new column 'Season' to the dataframe
df['Season'] = df['Date'] - pd.offsets.MonthBegin(7) + pd.DateOffset(months=1)
df['Season'] = df['Season'].dt.year

# Calculate the number of matches played by each team in each season
matches_home = df.groupby(['Season', 'HomeTeam'])['FTR'].count().reset_index()
matches_home.columns = ['Season', 'Team', 'Home_Matches']

matches_away = df.groupby(['Season', 'AwayTeam'])['FTR'].count().reset_index()
matches_away.columns = ['Season', 'Team', 'Away_Matches']

matches = pd.merge(matches_home, matches_away, on=['Season', 'Team'], how='outer').fillna(0)
matches['Matches'] = matches['Home_Matches'] + matches['Away_Matches']

# Calculate the number of wins and losses in each season for each team
wins_home = df[df['FTR'] == 'H'].groupby(['Season', 'HomeTeam'])['FTR'].count().reset_index()
wins_home.columns = ['Season', 'Team', 'Home_Wins']

wins_away = df[df['FTR'] == 'NH'].groupby(['Season', 'AwayTeam'])['FTR'].count().reset_index()
wins_away.columns = ['Season', 'Team', 'Away_Wins']

wins = pd.merge(wins_home, wins_away, on=['Season', 'Team'], how='outer').fillna(0)
wins['Wins'] = wins['Home_Wins'] + wins['Away_Wins']

matches_wins = pd.merge(matches, wins, on=['Season', 'Team'], how='left').fillna(0)
matches_wins['Losses'] = matches_wins['Matches'] - matches_wins['Wins']
matches_wins['Win_Percentage'] = matches_wins['Wins'] / matches_wins['Matches']


# Create a list of team names for the dropdown menu
team_list = matches_wins['Team'].unique().tolist()

# Allow the user to select which teams to display
selected_teams = st.multiselect('Select Teams', team_list)

# Filter the dataframe to include only the selected teams
filtered_data = matches_wins[matches_wins['Team'].isin(selected_teams)]


# Plot the win percentage for each team in each season using a line chart
fig = px.line(filtered_data, x='Season', y='Win_Percentage', color='Team', title='Win Percentage by Season')

# Show the plot using Streamlit
st.plotly_chart(fig)
