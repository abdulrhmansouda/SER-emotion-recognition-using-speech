import pandas as pd
from catboost import CatBoostClassifier, Pool
import os
import glob
import dictionary

curr_path = os.getcwd()

csv_path = os.path.join(curr_path,'features')

csvs = glob.glob(csv_path+'\\age_features*.csv')

for i,csv in enumerate(csvs):

    name_of_feature = csv.split('_')[-1].split('.')[0]
        
    # Load data from CSV file
    data = pd.read_csv(csv,header=None,error_bad_lines=False)
    
    target = data.shape[1] - 1

    # Separate features and target variable
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    # Create a Pool object for CatBoost
    pool = Pool(X, y)

    # Create a CatBoost classifier
    clf = CatBoostClassifier()

    # Select features using CatBoost feature importance
    feature_importances = clf.fit(pool).get_feature_importance()
    
    ## Note: 0.25 is not 25% because feature_importances > 1 or feature_importances < 0 , so it is not percent

    # 0.25    
    selected_features_25 = X.columns[feature_importances > 0.25]

    # 0.5
    selected_features_50 = X.columns[feature_importances > 0.5]

    #0.75
    selected_features_75 = X.columns[feature_importances > 0.75]

    # 1
    selected_features_100 = X.columns[feature_importances > 1]

    # convert selected features to list
    selected_features_25 = list(selected_features_25)
    selected_features_50 = list(selected_features_50)
    selected_features_75 = list(selected_features_75)
    selected_features_100 = list(selected_features_100)

    # merge togather
    selected_features = [selected_features_25,selected_features_50,selected_features_75,selected_features_100]
    # get names of columns
    DIC = dictionary.build_dic(name_of_feature)

    # create list of dictionries
    name_features = []
    
    # .25 .5 .75 1
    for i in range(4):
        # add dictionary
        name_features.append({})
        
        # save selected feature for each selected
        for feature in selected_features[i]:
            name_features[i][str(feature)] = DIC[feature]

        name_df = pd.DataFrame(name_features[i],index=[0])
        name_df.to_csv(csv_path+'\\NamesAgeFeaturesSelected_'+str(name_of_feature)+'_'+str((i+1)*25)+'.csv')

        # add target to features
        selected_features[i].append(target)

        # selected data 
        data1 = data[selected_features[i]]

        #save csv with selected 
        data1.to_csv(csv_path+'\\age_selected_features_'+str(name_of_feature)+'_'+str((i+1)*25)+'.csv')
