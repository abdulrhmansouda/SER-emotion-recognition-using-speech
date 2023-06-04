from sklearn.ensemble import GradientBoostingClassifier
import pickle
import helper
import time
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeatuersManagement")
from modules.FeatuersManagement.main import extract_feature_emotion_X_y_array

def get_classifier():
    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'GradientBoostingClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = para.test_size, random_state = 0)

        clf = GradientBoostingClassifier(learning_rate= 0.3, max_depth= 7, max_features= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 70, subsample= 0.7)

        clf.fit(X=X_train, y=y_train)
        pickle.dump(clf, open(helper.get_special_name('pickled', 'GradientBoostingClassifier', '.pickle'), "wb"))
    print('hello')
    return clf

def predict(path):
    classifier = get_classifier()
    return helper.predict(path,classifier)

if __name__ == '__main__':

    clf = get_classifier()

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = para.test_size, random_state = 0)
    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    y_prediction = clf.predict(X_test)
    helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction,classifier_name='GradientBoostingClassifier',title=f"Accuracy: {clf.score(X_test, y_test)} - GradientBoostingClassifier\n{helper.get_special_name(folder_name='',prefix='')}")