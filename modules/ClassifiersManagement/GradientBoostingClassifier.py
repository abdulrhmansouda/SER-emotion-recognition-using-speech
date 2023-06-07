from sklearn.ensemble import GradientBoostingClassifier
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
            'pickled', 'GradientBoostingClassifier', '.pickle'), "rb"))
    except:
        # print('hi')
        # exit()
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=para.test_size, random_state=0)

        clf = GradientBoostingClassifier(learning_rate=0.3, max_depth=7, max_features=None,
                                         min_samples_leaf=1, min_samples_split=2, n_estimators=70, subsample=0.7)

        clf.fit(X=X_train, y=y_train)
        pickle.dump(clf, open(helper.get_special_name(
            'pickled', 'GradientBoostingClassifier', '.pickle'), "wb"))
    print('hello')
    return clf


def get_classifier_through_randomized_search_cv():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV

    try:
        random_search = pickle.load(open(helper.get_special_name(
            'pickled', 'RandomizeSearch_GradientBoostingClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=para.test_size, random_state=0)

        # Define the parameter grid to search over
        param_dist = {
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
            'n_estimators': [40, 50, 70, 100, 200, 500],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [0.2, 0.5, 0.7, 2, 5, 10, 20],
            'min_samples_leaf': [0.2, 0.5, 1, 2, 4],
            'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None],
            'loss': ['deviance', 'exponential'],
            'random_state': [42]
        }

        # Create an MLPClassifier object
        clf = GradientBoostingClassifier()

        # Create a RandomizeSearchCV object
        random_search = RandomizedSearchCV(
            clf, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

        # Fit the grid search object to the data
        random_search.fit(X=X_train, y=y_train)
        pickle.dump(random_search, open(helper.get_special_name(
            'pickled', 'RandomizeSearch_GradientBoostingClassifier', '.pickle'), "wb"))
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
    helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_GradientBoostingClassifier',
                            title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_GradientBoostingClassifier\n{helper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")


# if __name__ == '__main__':

#     # print(1)
#     clf = get_classifier()
#     # print(2)
#     feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
#     X = feature_emotion_X_Y_array['X']
#     y = feature_emotion_X_Y_array['y']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = para.test_size, random_state = 0)


#     y_prediction = clf.predict(X_test)
#     helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction,classifier_name='GradientBoostingClassifier',title=f"Accuracy: {clf.score(X_test, y_test)} - GradientBoostingClassifier\n{helper.get_special_name(folder_name='',prefix='')}")
