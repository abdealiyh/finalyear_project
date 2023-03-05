# import pandas as pd
# import plotly.express as px

# # Read in the dataset
# df = pd.read_csv('ORIGINAL_FINAL.csv')

# # Convert Date column to datetime
# df['Date'] = pd.to_datetime(df['Date'])

# # Add a new column 'Season' to the dataframe
# df['Season'] = df['Date'] - pd.offsets.MonthBegin(7) + pd.DateOffset(months=1)
# df['Season'] = df['Season'].dt.year

# # Calculate the averages of various stats for each team and season
# averages = df.groupby(['Season', 'HomeTeam']).agg(
#     home_shots=('HS', 'mean'),
#     home_shots_on_target=('HST', 'mean'),
#     home_corners=('HC', 'mean'),
#     home_fouls_committed=('HF', 'mean'),
#     home_yellow_cards=('HY', 'mean'),
#     home_red_cards=('HR', 'mean')
# ).reset_index()

# averages = pd.merge(averages, df.groupby(['Season', 'AwayTeam']).agg(
#     away_shots=('AS', 'mean'),
#     away_shots_on_target=('AST', 'mean'),
#     away_corners=('AC', 'mean'),
#     away_fouls_committed=('AF', 'mean'),
#     away_yellow_cards=('AY', 'mean'),
#     away_red_cards=('AR', 'mean')
# ).reset_index(), left_on=['Season', 'HomeTeam'], right_on=['Season', 'AwayTeam'], how='outer')

# averages['Shots'] = averages['home_shots'] + averages['away_shots']
# averages['Shots_on_target'] = averages['home_shots_on_target'] + averages['away_shots_on_target']
# averages['Corners'] = averages['home_corners'] + averages['away_corners']
# averages['Fouls_committed'] = averages['home_fouls_committed'] + averages['away_fouls_committed']
# averages['Yellow_cards'] = averages['home_yellow_cards'] + averages['away_yellow_cards']
# averages['Red_cards'] = averages['home_red_cards'] + averages['away_red_cards']

# averages = averages[['Season', 'HomeTeam', 'Shots', 'Shots_on_target', 'Corners', 'Fouls_committed', 'Yellow_cards', 'Red_cards']]

# # Display the averages of stats for each team and season using a scatter plot
# fig = px.scatter(averages, x='Shots', y='Shots_on_target', size='Corners', color='Season', hover_data=['HomeTeam'])
# fig.show()



###########################################################
#######################################################
###########################################################


# import pandas as pd
# import plotly.express as px

# # Read in the dataset
# df = pd.read_csv('ORIGINAL_FINAL.csv')

# # Convert Date column to datetime
# df['Date'] = pd.to_datetime(df['Date'])

# # Add a new column 'Season' to the dataframe
# df['Season'] = df['Date'] - pd.offsets.MonthBegin(7) + pd.DateOffset(months=1)
# df['Season'] = df['Season'].dt.year

# # Group the data by team and season, and calculate the mean for each variable
# team_season = df.groupby(['Season', 'HomeTeam'])[['HS', 'HST', 'HC', 'HF', 'HY', 'HR']].mean().reset_index()
# team_season.columns = ['Season', 'Team', 'Shots', 'Shots_On_Target', 'Corners', 'Fouls_Committed', 'Yellow_Cards', 'Red_Cards']

# # Plot the variation of shots, shots on target, corners, fouls committed, yellow cards received and red cards received for each team over time
# fig = px.scatter(team_season, x='Season', y='Shots', size='Shots_On_Target', color='Team', hover_name='Team', log_y=True, size_max=40, title='Performance Metrics Over Time')
# fig.add_scatter(x=team_season['Season'], y=team_season['Corners'], mode='markers', marker=dict(size=team_season['Yellow_Cards'], sizemode='diameter', sizeref=0.1, sizemin=1), name='Corners')
# fig.add_scatter(x=team_season['Season'], y=team_season['Fouls_Committed'], mode='markers', marker=dict(size=team_season['Red_Cards'], sizemode='diameter', sizeref=0.1, sizemin=1), name='Fouls Committed')
# fig.show()


###########################################################
#######################################################
###########################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Read in the dataset
df = pd.read_csv('ORIGINAL_FINAL.csv')

# Create a dropdown menu to select the team to display
n = 5700
df2 = df[n:]
#team_names = df2['HomeTeam'].unique()

teams = sorted(list(df2['HomeTeam'].unique()))
team1 = st.selectbox('Select Team 1', teams)
available_away_teams = [team for team in teams if team != team1]
team2 = st.selectbox('Select Team 2', available_away_teams)


# Create a slider widget to select the number of seasons
seasons = st.slider('Number of seasons', min_value=1, max_value=16, value=3)


# Filter the data for the selected teams and number of seasons
team1_data = df[(df['HomeTeam'] == team1) | (df['AwayTeam'] == team1)]
team2_data = df[(df['HomeTeam'] == team2) | (df['AwayTeam'] == team2)]

team1_data = team1_data.tail(seasons * 38)
team2_data = team2_data.tail(seasons * 38)


# Calculate the total values for each feature
team1_totals = [team1_data['HS'].sum(), team1_data['HST'].sum(), team1_data['HC'].sum(),
                team1_data['HF'].sum(), team1_data['HY'].sum(), team1_data['HR'].sum()]
team2_totals = [team2_data['HS'].sum(), team2_data['HST'].sum(), team2_data['HC'].sum(),
                team2_data['HF'].sum(), team2_data['HY'].sum(), team2_data['HR'].sum()]

# # Create a dataframe for the team statistics
# data = pd.DataFrame({'Features': ['Shots', 'Shots on Target', 'Corners', 'Fouls', 'Yellow Cards', 'Red Cards'],
#                      'Team 1': team1_totals,
#                      'Team 2': team2_totals})

# # Melt the dataframe for plotting
# melted_data = pd.melt(data, id_vars=['Features'], var_name='Team', value_name='Totals')

# # Create a bar chart using plotly.express
# fig = px.bar(melted_data, x='Features', y='Totals', color='Team',
#              title='Team Statistics Comparison')

# # Update the figure layout
# fig.update_layout(xaxis=dict(title='Features'), yaxis=dict(title='Totals'))

###############################################################################################
###############################################################################################

# Create the bar chart using plotly.graph_objs
data = [go.Bar(x=['Shots', 'Shots on Target', 'Corners', 'Fouls Committed', 'Yellow Cards', 'Red Cards'], 
               y=team1_totals, name=team1),
        go.Bar(x=['Shots', 'Shots on Target', 'Corners', 'Fouls Committed', 'Yellow Cards', 'Red Cards'], 
               y=team2_totals, name=team2)]

# Set the layout for the bar chart
layout = go.Layout(title='Team Statistics Comparison', 
                   xaxis=dict(title='Features'),
                   yaxis=dict(title='Total Values'))




# Create the figure using the data and layout
fig = go.Figure(data=data, layout=layout)



# Show the figure
st.plotly_chart(fig)

