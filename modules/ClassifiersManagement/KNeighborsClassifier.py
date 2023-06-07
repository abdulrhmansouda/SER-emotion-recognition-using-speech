from sklearn.neighbors import KNeighborsClassifier
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
            'pickled', 'KNeighborsClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=para.test_size, random_state=0)

        clf = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance')

        clf.fit(X=X_train, y=y_train)
        pickle.dump(clf, open(helper.get_special_name(
            'pickled', 'KNeighborsClassifier', '.pickle'), "wb"))
    return clf


def get_classifier_through_randomized_search_cv():
    from sklearn.model_selection import RandomizedSearchCV

    try:
        random_search = pickle.load(open(helper.get_special_name(
            'pickled', 'RandomizeSearch_KNeighborsClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=para.test_size, random_state=0)

        # Define the parameter grid to search over
        param_dist = {
            'n_neighbors': [3, 5, 10, 20, 50],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 50],
            'p': [1, 2, 3, 4, 5],
            'n_jobs': [-1],
        }

        # Create an MLPClassifier object
        clf = KNeighborsClassifier()

        # Create a RandomizeSearchCV object
        random_search = RandomizedSearchCV(
            clf, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

        # Fit the grid search object to the data
        random_search.fit(X=X_train, y=y_train)
        pickle.dump(random_search, open(helper.get_special_name(
            'pickled', 'RandomizeSearch_KNeighborsClassifier', '.pickle'), "wb"))
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
    helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_KNeighborsClassifier',
                            title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_KNeighborsClassifier\n{helper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")


# if __name__ == '__main__':

#     clf = get_classifier()

#     feature_emotion_X_Y_array_for_test = get_feature_emotion_X_Y_array_for(
#         'test')

#     X_test = feature_emotion_X_Y_array_for_test['X'].to_numpy()[:, 1:]
#     y_test = feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:, 1:]

#     # exit()
#     predicted_array_Y = clf.predict(X_test)
#     confusion_matrix(y_test=y_test, y_prediction=predicted_array_Y,
#                      title=f"Accuracy: {clf.score(X_test, y_test)} - KNeighborsClassifier\n{helper.get_special_name(folder_name='',prefix='')}")
