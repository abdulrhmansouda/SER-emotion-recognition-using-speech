from sklearn.ensemble import RandomForestClassifier
import pickle
import helper
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array


def get_classifier():
    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'RandomForestClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=para.test_size, random_state=0)

        clf = RandomForestClassifier(
            max_depth=7, max_features=0.5, min_samples_leaf=1, min_samples_split=2, n_estimators=40)

        clf.fit(X=X_train, y=y_train)
        pickle.dump(clf, open(helper.get_special_name(
            'pickled', 'RandomForestClassifier', '.pickle'), "wb"))
    return clf


def get_classifier_through_randomized_search_cv():
    from sklearn.model_selection import RandomizedSearchCV

    try:
        random_search = pickle.load(open(helper.get_special_name(
            'pickled', 'RandomizeSearch_RandomForestClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=para.test_size, random_state=0)

        # Define the parameter grid to search over
        param_dist = {
            'n_estimators': [10, 50, 100, 500],
            'max_depth': [None, 3, 5, 10, 20],
            'min_samples_split': [0.2, 0.5, 0.7, 2, 5, 10, 20],
            'min_samples_leaf': [0.2, 0.5, 1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'n_jobs': [-1],
            'random_state': [42]
        }

        # Create an RandomForestClassifier object
        clf = RandomForestClassifier()

        # Create a RandomizeSearchCV object
        random_search = RandomizedSearchCV(
            clf, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

        # Fit the grid search object to the data
        random_search.fit(X=X_train, y=y_train)
        pickle.dump(random_search, open(helper.get_special_name(
            'pickled', 'RandomizeSearch_RandomForestClassifier', '.pickle'), "wb"))
    # Print the best parameters and best score
    print("Best parameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)

    return random_search


def predict(path):
    classifier = get_classifier()
    return helper.predict(path, classifier)


if __name__ == '__main__':
    # clf = get_classifier()
    clf = get_classifier_through_randomized_search_cv()

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=para.test_size, random_state=0)

    y_prediction = clf.predict(X_test)
    helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_RandomForestClassifier',
                            title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_RandomForestClassifier\n{helper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")


# if __name__ == '__main__':

#     clf = get_classifier()

#     feature_emotion_X_Y_array_for_test = get_feature_emotion_X_Y_array_for(
#         'test')

#     X_test = feature_emotion_X_Y_array_for_test['X'].to_numpy()[:, 1:]
#     y_test = feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:, 1:]

#     # exit()
#     predicted_array_Y = clf.predict(X_test)
#     confusion_matrix(y_test=y_test, y_prediction=predicted_array_Y,
#                      title=f"Accuracy: {clf.score(X_test, y_test)} - RandomForestClassifier\n{helper.get_special_name(folder_name='',prefix='')}")
