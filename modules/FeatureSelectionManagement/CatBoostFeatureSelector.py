# from modules.FeaturesManagement.Feature import Feature
import pickle
import sys
import os
sys.path.insert(0, os.getcwd())
# import parameters as para
# sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
# import modules.ClassifiersManagement.helper as classifierHelper
# sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
# from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array

class CatBoostFeatureSelector:
    def get_selector():
        sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
        import modules.ClassifiersManagement.helper as classifierHelper
        import catboost as cb
        # import pandas as pd
        try:
            clf = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'CatBoostFeatureSelector', '.pickle'), "rb"))
        except:
            sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
            from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array(filter=False)
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']

            # Create a CatBoostClassifier model
            clf = cb.CatBoostClassifier(
                                        # iterations=500
                                        # ,learning_rate=0.01,
                                        depth=5,
                                        )
            # print(y)
            # exit()
            # Fit the model to the data
            clf.fit(X, y)
            pickle.dump(clf, open(classifierHelper.get_special_name(
                'pickled', 'CatBoostFeatureSelector', '.pickle'), "wb"))
        # print('hello')
        return clf
    
    def get_feature_importance():
        model = CatBoostFeatureSelector.get_selector()
        feature_importance = model.get_feature_importance()
        return feature_importance
        # feature_importance > max(feature_importance)*0.5

    def get_boolean_array_for_most_important_feature(ratio=0.1):
        feature_importance = CatBoostFeatureSelector.get_feature_importance()
        return feature_importance >= max(feature_importance)*ratio
    
    def filter_features(X):
        import numpy as np
        # Convert the input list to a numpy array
        X = np.array(X)
        # print(X.shape)
        y = CatBoostFeatureSelector.get_boolean_array_for_most_important_feature()
        # print(y.shape)
        # exit()
        return X[:,y]

if __name__ == '__main__':
    sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
    import modules.ClassifiersManagement.helper as classifierHelper
    import catboost as cb
    # import pandas as pd
    # print(CatBoostFeatureSelector.filter_features([[1,2,3,4,5,6,7,8,9,10,11],[21,22,23,24,25,26,27,28,29,210,211]]))
    sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
    from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array
    print('hi')
    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array(filter=False)
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    print('hi')
    # Create a CatBoostClassifier model
    clf = cb.CatBoostClassifier()

    # Fit the model to the data
    clf.fit(X, y)
    pickle.dump(clf, open(classifierHelper.get_special_name('pickled', 'CatBoostFeatureSelector', '.pickle'), "wb"))