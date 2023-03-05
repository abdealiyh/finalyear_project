import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np



from sklearn.preprocessing import StandardScaler


from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)

def load_data(df):
    df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)
    
    le = LabelEncoder()
    df['FTR'] = le.fit_transform(df['FTR'])
    df['FTR'] = 1 - df['FTR']
    

    # def calculate_ShAcc(HS, AS, HST, AST):
    #     if HS > 0:
    #         home_ShAcc = (HST / HS)
    #     else:
    #         home_ShAcc = 0
    
    #     if AS > 0:  
    #         away_ShAcc = (AST / AS) 
    #     else:
    #         away_ShAcc = 0
            
    #     return home_ShAcc, away_ShAcc
    
    # df['Home_ShAcc'], df['Away_ShAcc'] = zip(*df.apply(lambda row: calculate_ShAcc(row['HS'], row['AS'],  row['HST'], row['AST']), axis=1))
    
    def calculate_elo_rating(df, K=40, init_rating = 1500):
        teams = set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist())
        team_ratings = {team: init_rating for team in teams}
    
        for index, row in df.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            home_team_rating = team_ratings[home_team] 
            away_team_rating = team_ratings[away_team] 
        
            expected_outcome = 1 / (1 + 10 ** ((away_team_rating - home_team_rating) / 400))
            if row["FTHG"] > row["FTAG"]:
                result = 1
            else:
                result = 0
            team_ratings[home_team] = home_team_rating + K * (result - expected_outcome)
            team_ratings[away_team] = away_team_rating + K * (1 - result - (1 - expected_outcome))
            df["HomeTeamElo"] = [team_ratings[team] for team in df["HomeTeam"]]
            df["AwayTeamElo"] = [team_ratings[team] for team in df["AwayTeam"]]
        return df

    df = calculate_elo_rating(df)
    return df



def train_models(df):
   
    n = 5700
    df2 = df[n:]
    n2 = 5700
    df3 = df[:n2]

    # Filter out non-object columns
    non_object_cols = [col for col in df.columns if df[col].dtype != "object"]
    
    X_train = df3[non_object_cols]
    X_test = df2[non_object_cols]
    y_test = df2["FTR"]
    y_train = df3["FTR"]

    X_train = X_train.drop(['FTHG','FTAG','FTR','MW'], axis=1)
    X_test = X_test.drop(['FTHG','FTAG','FTR','MW'], axis=1)


    #Train an XGBoost model
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return xgb


def main():
    
    df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)
    
    df = load_data(df)
    st.dataframe(df)
    xgb = train_models(df)

    xgb = XGBClassifier()

    n = 5700
    df2 = df[n:]
    n2 = 5700
    df3 = df[:n2]

    non_object_cols = [col for col in df.columns if df[col].dtype != "object"]


    X_train = df3[non_object_cols]
    X_test = df2[non_object_cols]
    y_test = df2["FTR"]
    y_train = df3["FTR"]

    X_train = X_train.drop(['FTHG','FTAG','FTR','MW'], axis=1)
    X_test = X_test.drop(['FTHG','FTAG','FTR','MW'], axis=1)

    
    # Train a Logistic Regression model
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)


    # Create the select boxes for the home and away teams
    team_names = df['HomeTeam'].unique()
    home_team = st.selectbox("Select the home team:", team_names)
    away_team = st.selectbox("Select the away team:", team_names)

   # Create the input boxes for the features
    home_gs = st.number_input("Home team Goals Scored", value=0)
    away_gs = st.number_input("Away team Goals Scored", value=0)
    home_gc = st.number_input("Home team Goals Conceded", value=0)
    away_gc = st.number_input("Away team Goals Conceded", value=0)
    home_points = st.number_input("Home team Points", value=0.0, step=0.1)
    away_points = st.number_input("Away team Points", value=0.0, step=0.1)
    home_form = st.number_input("Home team Form Points", value=0)
    away_form = st.number_input("Away team Form Points", value=0)
    home_win3 = st.number_input("Home team WinStreak 3", value=0)
    home_win5 = st.number_input("Home team WinStreak 5", value=0)
    home_loss3 = st.number_input("Home team LossStreak 3", value=0)
    home_loss5 = st.number_input("Home team LossStreak 5", value=0)
    away_win3 = st.number_input("Away team WinStreak 3", value=0)
    away_win5 = st.number_input("Away team WinStreak 5", value=0)
    away_loss3 = st.number_input("Away team LossStreak 3", value=0)
    away_loss5 = st.number_input("Away team LossStreak 5", value=0)
    home_gd = st.number_input("Home team Goal Difference", value=0.0, step=0.1)
    away_gd = st.number_input("Away team Goal Difference", value=0.0, step=0.1)
    diff_pts = st.number_input("Points Difference", value=0.0 , step=0.1)
    diff_formpts = st.number_input("Form Points Difference", value=0.0, step=0.1)
    home_elo = st.number_input("Home Team ELO Rating", value=0.0, step=0.1)
    away_elo = st.number_input("Away Team ELO Rating", value=0.0, step=0.1)



    # Add a predict button
    if st.button("Predict"):
        new_data = {'HomeTeam': [home_team], 'AwayTeam': [away_team],'HTGS': [home_gs], 'ATGS': [away_gs],
        'HTGC': [home_gc], 'ATGC': [away_gc],'HTP': [home_points], 'ATP': [away_points],
        'HTFormPts': [home_form], 'ATFormPts': [away_form],'HTWinStreak3': [home_win3], 'HTWinStreak5': [home_win5],
        'HTLossStreak3': [home_loss3], 'HTLossStreak5': [home_loss5],'ATWinStreak3': [away_win3], 'ATWinStreak5': [away_win5],
        'ATLossStreak3': [away_loss3], 'ATLossStreak5': [away_loss5],'HTGD': [home_gd], 'ATGD': [away_gd],
        'DiffPts': [diff_pts], 'DiffFormPts': [diff_formpts], 'HomeTeamElo': [home_elo], 'AwayTeamElo': [away_elo]}

        # new_df = pd.DataFrame(new_data, columns=['HomeTeam', 'AwayTeam','HTGS','ATGS', 'HTGC','ATGC','HTP', 'ATP','HTFormPts',
        # 'ATFormPts' ,'HTWinStreak3','HTWinStreak5', 'HTLossStreak3','HTLossStreak5','ATWinStreak3', 'ATWinStreak5',
        # 'ATLossStreak3','ATLossStreak5', 'HTGD','ATGD','DiffPts','DiffFormPts','HomeTeamElo','AwayTeamElo'])

        # merged_df = pd.merge(new_df, df2, on=['HomeTeam', 'AwayTeam'], how='left')

        feature_names = X_train.columns.tolist()
        new_df = pd.DataFrame(new_data, columns=feature_names)
        # new_df = new_df.drop(['HomeTeam', 'AwayTeam'], axis=1)

        # Filter out non-object columns
        # non_object_cols = [col for col in df.columns if df[col].dtype != "object"]
        # merged_df2 = merged_df[non_object_cols]
        # new_df = new_df.drop(['HomeTeam', 'AwayTeam'], axis=1)

        # actual_results = merged_df["FTR"].values

        # if actual_results == 1:
        #     actual_results=home_team
        # else:
        #     actual_results=away_team
        

        prediction = xgb.predict(new_df)
        
        
        
        if prediction == 1:
            prediction=home_team
        else:
            prediction=away_team

        st.success("The predicted winner is: {}".format(prediction))

        # st.success("The actual winner is: {}".format(actual_results))

        # st.success("The acuracy of model is: {:.2f}%".format(accuracy * 100))
        


# Run the app
if __name__ == '__main__':
    main()
