# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import GridSearchCV
# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np

# # from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import StandardScaler

# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.svm import SVC
# # from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, f1_score, make_scorer
# from sklearn.preprocessing import LabelEncoder


# df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)

# def load_data(df):
#     df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)
#     le = LabelEncoder()
#     df['FTR'] = le.fit_transform(df['FTR'])
#     df['FTR'] = 1 - df['FTR']
    

#     # def calculate_ShAcc(HS, AS, HST, AST):
#     #     if HS > 0:
#     #         home_ShAcc = (HST / HS)
#     #     else:
#     #         home_ShAcc = 0
    
#     #     if AS > 0:  
#     #         away_ShAcc = (AST / AS) 
#     #     else:
#     #         away_ShAcc = 0
            
#     #     return home_ShAcc, away_ShAcc
    
#     # df['Home_ShAcc'], df['Away_ShAcc'] = zip(*df.apply(lambda row: calculate_ShAcc(row['HS'], row['AS'],  row['HST'], row['AST']), axis=1))
    
#     def calculate_elo_rating(df, K=40, init_rating = 1500):
#         teams = set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist())
#         team_ratings = {team: init_rating for team in teams}
    
#         for index, row in df.iterrows():
#             home_team = row["HomeTeam"]
#             away_team = row["AwayTeam"]
#             home_team_rating = team_ratings[home_team] 
#             away_team_rating = team_ratings[away_team] 
        
#             expected_outcome = 1 / (1 + 10 ** ((away_team_rating - home_team_rating) / 400))
#             if row["FTHG"] > row["FTAG"]:
#                 result = 1
#             else:
#                 result = 0
#             team_ratings[home_team] = home_team_rating + K * (result - expected_outcome)
#             team_ratings[away_team] = away_team_rating + K * (1 - result - (1 - expected_outcome))
#             df["HomeTeamElo"] = [team_ratings[team] for team in df["HomeTeam"]]
#             df["AwayTeamElo"] = [team_ratings[team] for team in df["AwayTeam"]]
#         return df

#     df = calculate_elo_rating(df)
#     return df

# # df = pd.read_csv('FINAL_MERGED_FINAL_DATASET.csv', parse_dates=['Date'], index_col=0)
# # df = load_data(df)

# def train_models(df):
#     #df = load_data(df)
#     n = 5700
#     df2 = df[n:]
#     n2 = 5700
#     df3 = df[:n2]

#     # Filter out non-object columns
#     non_object_cols = [col for col in df.columns if df[col].dtype != "object"]
#     #X = df[non_object_cols]
#     #y = df["FTR"]
#     #X= X.drop(['HTHG','HTAG','FTHG','FTAG','FTR','MW'], axis=1)

#     X_train = df3[non_object_cols]
#     X_test = df2[non_object_cols]
#     y_test = df2["FTR"]
#     y_train = df3["FTR"]

#     X_train = X_train.drop(['FTHG','FTAG','FTR','MW'], axis=1)
#     X_test = X_test.drop(['FTHG','FTAG','FTR','MW'], axis=1)

#     # Split the data into training and testing sets
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # # Train a Logistic Regression model
#     # lr = LogisticRegression()
#     # lr.fit(X_train, y_train)

#     # # Train a Decision Tree Classifier model
#     # dt = DecisionTreeClassifier()
#     # dt.fit(X_train, y_train)

#     # # Train a Random Forest Classifier model
#     # rf = RandomForestClassifier()
#     # rf.fit(X_train, y_train)

#     # # Train a Support Vector Machine (SVM) model
#     # svm = SVC()
#     # svm.fit(X_train, y_train)

#     # # Train a K-Nearest Neighbors (KNN) model
#     # knn = KNeighborsClassifier()
#     # knn.fit(X_train, y_train)

#     #Train an XGBoost model
#     xgb = XGBClassifier()
#     xgb.fit(X_train, y_train)

#     # # Evaluate the models
#     # models = [xgb]
#     # model_names = ["XGBoost"]

    
#     y_pred = xgb.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     return accuracy



    



# def calculate_elo_rating(df, K=40, init_rating = 1500):
#     df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)
#     le = LabelEncoder()
#     df['FTR'] = le.fit_transform(df['FTR'])
#     df['FTR'] = 1 - df['FTR']

#     teams = set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist())
#     team_ratings = {team: init_rating for team in teams}
    
#     for index, row in df.iterrows():
#         home_team = row["HomeTeam"]
#         away_team = row["AwayTeam"]
#         home_team_rating = team_ratings[home_team] 
#         away_team_rating = team_ratings[away_team] 
        
#         expected_outcome = 1 / (1 + 10 ** ((away_team_rating - home_team_rating) / 400))
#         if row["FTHG"] > row["FTAG"]:
#             result = 1
#         else:
#             result = 0
#         team_ratings[home_team] = home_team_rating + K * (result - expected_outcome)
#         team_ratings[away_team] = away_team_rating + K * (1 - result - (1 - expected_outcome))
#         df["HomeTeamElo"] = [team_ratings[team] for team in df["HomeTeam"]]
#         df["AwayTeamElo"] = [team_ratings[team] for team in df["AwayTeam"]]
#     return df






# # home_team = input('Enter the home team name: ')
# # away_team = input('Enter the away team name: ')

# # new_data = {'HomeTeam': [home_team], 'AwayTeam': [away_team]}

# # new_df = pd.DataFrame(new_data, columns=['HomeTeam', 'AwayTeam'])

# # merged_df = pd.merge(new_df, df, on=['HomeTeam', 'AwayTeam'], how='left')

# # # Filter out non-object columns
# # non_object_cols = [col for col in df.columns if df[col].dtype != "object"]
# # merged_df2 = merged_df[non_object_cols]
# # merged_df2 = merged_df2.drop(['HTHG','HTAG','FTHG','FTAG','FTR','MW'], axis=1)

# # actual_results = merged_df["FTR"].values

# # prediction = dt.predict(merged_df2)
# # num_correct_predictions = sum([1 for i in range(len(prediction)) if prediction[i] == actual_results[i]])
# # accuracy = num_correct_predictions / len(prediction)
# # accuracy

# # prediction = rf.predict(merged_df2)
# # num_correct_predictions = sum([1 for i in range(len(prediction)) if prediction[i] == actual_results[i]])
# # accuracy = num_correct_predictions / len(prediction)
# # accuracy

# # prediction = svm.predict(merged_df2)
# # num_correct_predictions = sum([1 for i in range(len(prediction)) if prediction[i] == actual_results[i]])
# # accuracy = num_correct_predictions / len(prediction)
# # accuracy

# # prediction = knn.predict(merged_df2)
# # num_correct_predictions = sum([1 for i in range(len(prediction)) if prediction[i] == actual_results[i]])
# # accuracy = num_correct_predictions / len(prediction)
# # accuracy

# # prediction = xgb.predict(merged_df2)
# # num_correct_predictions = sum([1 for i in range(len(prediction)) if prediction[i] == actual_results[i]])
# # accuracy = num_correct_predictions / len(prediction)
# # accuracy


# def main():


#     #df = pd.read_csv('FINAL_MERGED_FINAL_DATASET.csv', parse_dates=['Date'], index_col=0)
#     df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)
#     #df = load_data(df)
#     #df = calculate_elo_rating(df)
#     #accuracy = train_models(df)
#     def calculate_elo_rating(df, K=40, init_rating = 1500):
#         #df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)
#         le = LabelEncoder()
#         df['FTR'] = le.fit_transform(df['FTR'])
#         df['FTR'] = 1 - df['FTR']

#         teams = set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist())
#         team_ratings = {team: init_rating for team in teams}
            
#         for index, row in df.iterrows():
#             home_team = row["HomeTeam"]
#             away_team = row["AwayTeam"]
#             home_team_rating = team_ratings[home_team] 
#             away_team_rating = team_ratings[away_team] 
                
#             expected_outcome = 1 / (1 + 10 ** ((away_team_rating - home_team_rating) / 400))
#             if row["FTHG"] > row["FTAG"]:
#                 result = 1
#             else:
#                 result = 0
#             team_ratings[home_team] = home_team_rating + K * (result - expected_outcome)
#             team_ratings[away_team] = away_team_rating + K * (1 - result - (1 - expected_outcome))
#             df["HomeTeamElo"] = [team_ratings[team] for team in df["HomeTeam"]]
#             df["AwayTeamElo"] = [team_ratings[team] for team in df["AwayTeam"]]
#         return df

#     df = calculate_elo_rating(df) 
#     # xgb = XGBClassifier()

#     n = 5700
#     df2 = df[n:]
#     n2 = 5700
#     df3 = df[:n2]

#     non_object_cols = [col for col in df.columns if df[col].dtype != "object"]


#     X_train = df3[non_object_cols]
#     X_test = df2[non_object_cols]
#     y_test = df2["FTR"]
#     y_train = df3["FTR"]

#     X_train = X_train.drop(['FTHG','FTAG','FTR','MW'], axis=1)
#     X_test = X_test.drop(['FTHG','FTAG','FTR','MW'], axis=1)

#     # Split the data into training and testing sets
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Train a Logistic Regression model
#     # xgb = XGBClassifier()
#     # xgb.fit(X_train, y_train)

#     # Define the parameter grid
#     parameters = { 'learning_rate' : [0.01],
#                 'n_estimators' : [500],
#                 'max_depth': [10],
#                 'min_child_weight': [10],
#                 'gamma':[0.1],
#                 'subsample' : [0.4],
#                 'colsample_bytree' : [0.4],
#                 'scale_pos_weight' : [1],
#                 'reg_alpha':[1e-5]
#                 }
    
#     def predict_labels(clf, features, target):
#         ''' Makes predictions using a fit classifier based on F1 score. '''

#         y_pred = clf.predict(features)
        
#         return f1_score(target, y_pred, pos_label=1), sum(target == y_pred) / float(len(y_pred))


#     # TODO: Initialize the classifier
#     clf = xgb.XGBClassifier(seed=1)

#     # TODO: Make an f1 scoring function using 'make_scorer' 
#     f1_scorer = make_scorer(f1_score,pos_label=1)

#     # TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
#     grid_obj = GridSearchCV(clf,
#                             scoring=f1_scorer,
#                             param_grid=parameters,
#                             cv=5)

#     # TODO: Fit the grid search object to the training data and find the optimal parameters
#     grid_obj = grid_obj.fit(X_train,y_train)

#     # Get the estimator
#     clf = grid_obj.best_estimator_
#     print(clf)

#     # Report the final F1 score for training and testing after parameter tuning
#     f1, acc = predict_labels(clf, X_train, y_train)
#     #st.write( "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
        
#     f1, acc = predict_labels(clf, X_test, y_test)
#     #st.write("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

#     # y_pred = xgb.predict(X_test)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     #st.write("The accuracy of model is: {:.2f}%".format(accuracy * 100))

#     # Create the select boxes for the home and away teams
#     team_names = df['HomeTeam'].unique()
#     home_team = st.selectbox("Select the home team:", team_names)
#     away_team = st.selectbox("Select the away team:", team_names)

#     # home_team = st.text_input("Enter Home Team Name: ")
#     # away_team = st.text_input("Enter Away Team Name: ")

#     # Add a predict button
#     if st.button("Predict"):
#         new_data = {'HomeTeam': [home_team], 'AwayTeam': [away_team]}

#         new_df = pd.DataFrame(new_data, columns=['HomeTeam', 'AwayTeam'])

#         merged_df = pd.merge(new_df, df2, on=['HomeTeam', 'AwayTeam'], how='left')

#         # Filter out non-object columns
#         non_object_cols = [col for col in df.columns if df[col].dtype != "object"]
#         merged_df2 = merged_df[non_object_cols]
#         merged_df2 = merged_df2.drop(['FTHG','FTAG','FTR','MW'], axis=1)

#         actual_results = merged_df["FTR"].values

#         if actual_results == 1:
#             actual_results=home_team
#         else:
#             actual_results=away_team
#         #st.write(actual_results)

#         prediction = xgb.predict(merged_df2)
#         # num_correct_predictions = sum([1 for i in range(len(prediction)) if prediction[i] == actual_results[i]])
#         # accuracy = num_correct_predictions / len(prediction)
        
#         if prediction == 1:
#             prediction=home_team
#         else:
#             prediction=away_team

#         st.success("The predicted winner is: {}".format(prediction))

#         st.success("The actual winner is: {}".format(actual_results))

#         st.success("The accuracy of model is: {:.2f}%".format(accuracy * 100))
#         # st.write("Best model: {}".format(best_model))
#         # st.write("Accuracy: {:.2f}%".format(best_accuracy * 100))


# # Run the app
# if __name__ == '__main__':
#     main()



######################################################################################################################
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def load_data():
    df = pd.read_csv('final_dataset_final.csv', parse_dates=['Date'], index_col=0)
    le = LabelEncoder()
    df['FTR'] = le.fit_transform(df['FTR'])
    df['FTR'] = 1 - df['FTR']
    return df

def calculate_elo_rating(df, K=40, init_rating=1500):

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

def train_model(df):
    df = calculate_elo_rating(df)
    xgb = XGBClassifier(colsample_bytree=0.4, gamma=0.1, learning_rate=0.01, max_depth=10,
              min_child_weight=10, n_estimators=500, reg_alpha=1e-05, seed=1,
              subsample=0.4)
    
    n2 = 5700
    df3 = df[:n2]

    non_object_cols = [col for col in df.columns if df[col].dtype != "object"]

    X_train = df3[non_object_cols]
    y_train = df3["FTR"]

    X_train = X_train.drop(['FTHG','FTAG','FTR','MW'], axis=1)
    X_train = df[non_object_cols].drop(['FTHG','FTAG','FTR','MW'], axis=1)
    y_train = df["FTR"]

    xgb.fit(X_train, y_train)

    return xgb

def predict_winner(xgb, home_team, away_team, df):
    df = calculate_elo_rating(df)
    n = 5700
    df2 = df[n:]
    new_data = {'HomeTeam': [home_team], 'AwayTeam': [away_team]}
    new_df = pd.DataFrame(new_data, columns=['HomeTeam', 'AwayTeam'])
    merged_df = pd.merge(new_df, df2, on=['HomeTeam', 'AwayTeam'], how='left')
    non_object_cols = [col for col in df.columns if df[col].dtype != "object"]
    merged_df2 = merged_df[non_object_cols]
    merged_df2 = merged_df2.drop(['FTHG','FTAG','FTR','MW'], axis=1)
    prediction = xgb.predict(merged_df2)[0]
    actual_results = merged_df["FTR"].values
    return prediction, actual_results

def main():
    st.title("Premier League Match Predictor")

    # Load data
    df = load_data()

    # Train the model
    xgb = train_model(df)

    # # Create the select boxes for the home and away teams
    # team_names = df['HomeTeam'].unique()
    # home_team = st.selectbox("Select the home team:", team_names)
    # away_team = st.selectbox("Select the away team:", team_names)


    # Create the select boxes for the home and away teams
    #team_names = ['Team 1', 'Team 2', 'Team 3', ..., 'Team 20']
    n = 5700
    df2 = df[n:]
    team_names = df2['HomeTeam'].unique()
    home_team = st.selectbox("Select the home team:", team_names)
    available_away_teams = [team for team in team_names if team != home_team]
    away_team = st.selectbox("Select the away team:", available_away_teams)

    # Add a predict button
    if st.button("Predict"):
        prediction, actual_results = predict_winner(xgb, home_team, away_team, df)
        #actual_result = df.loc[(df['HomeTeam']==home_team) & (df['AwayTeam']==away_team), 'FTR'].values[0]
        

        # if prediction == 1:
        #     prediction=home_team
        # else:
        #     prediction=away_team

        # # if actual_results == 1:
        # #     actual_winner = home_team
        # # else:
        # #     actual_winner = away_team

        # # Display the predicted winner and actual winner
        # st.success("The predicted winner is: {}".format(prediction))
        # # st.success("The actual winner is: {}".format(actual_winner))





        # Define CSS style for animation and colors
        winner_style = """
            <style>
                .winner {
                    animation-name: example;
                    animation-duration: 2s;
                    animation-iteration-count: infinite;
                    font-size: 36px;
                    text-align: center;
                    margin-top: 20px;
                    margin-bottom: 20px;
                }
                @keyframes example {
                    0%   {color: green; text-shadow: 2px 2px white;}
                    25%  {color: blue; text-shadow: 2px 2px white;}
                    50%  {color: red; text-shadow: 2px 2px white;}
                    75%  {color: orange; text-shadow: 2px 2px white;}
                    100% {color: purple; text-shadow: 2px 2px white;}
                }
            </style>
        """

        # Add the style to the Streamlit page
        st.markdown(winner_style, unsafe_allow_html=True)

        # Display the predicted winner in an animated and colored way
        if prediction == 1:
            prediction=home_team
            st.markdown(f'<p class="winner">{prediction} wins!</p>', unsafe_allow_html=True)
        else:
            prediction=away_team
            st.markdown(f'<p class="winner">{prediction} wins!</p>', unsafe_allow_html=True)










# Run the app
if __name__ == '__main__':
    main()