from sklearn.svm import SVC
import svm 
import pickle
import helper
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeatuersManagement")
from modules.FeatuersManagement.main import extract_feature_emotion_X_y_array
from modules.FeatuersManagement.Feature import Feature

def get_classifier():
    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'SVCClassifier', '.pickle'), "rb"))
        from joblib import dump, load
        sc=load(open(helper.get_special_name(
            'pickled', 'sc_SVCClassifier', '.pickle'), "rb"))
    except:
        print('here')
        feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=para.test_size, random_state=0)
        # Feature Scaling
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        # clf = SVC(C=3, gamma=0.01, kernel='rbf',
        #           random_state=0, max_iter=100000)

        # clf.fit(X=X_train, y=y_train)
        clf,sc = svm.train(X,y,3,'abd')
        from joblib import dump, load
        # from sklearn.externals.joblib import dump, load
        dump(sc, open(helper.get_special_name(
            'pickled', 'sc_SVCClassifier', '.bin'), "wb"))
        pickle.dump(clf, open(helper.get_special_name(
            'pickled', 'SVCClassifier', '.pickle'), "wb"))
    print('hello')
    return clf,sc


# def predict(path):
#     from sklearn.preprocessing import StandardScaler
#     sc = StandardScaler()
#     feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
#     X = feature_emotion_X_Y_array['X']
#     y = feature_emotion_X_Y_array['y']
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=para.test_size, random_state=0)
#     # from sklearn.preprocessing import StandardScaler
#     # sc = StandardScaler()
#     # X_train = sc.fit_transform(X_train)
#     # feature = sc.transform(Feature(path).extract_feature().reshape(1, -1))
#     # classifier = get_classifier()
#     return para.emotions[int(classifier.predict(feature)[0])]

def predict(path):
    classifier,sc = get_classifier()
    return helper.predict(path,classifier,sc)

# predict('hi')

if __name__ == '__main__':

    clf ,sc= get_classifier()

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=para.test_size, random_state=0)
    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    y_prediction = clf.predict(X_test)
    # confusion_matrix(y_test=y_test, y_prediction=y_prediction,title=f"Accuracy: {clf.score(X_test, y_test)} - SVC\n{helper.get_special_name(folder_name='',prefix='')}")
    helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='SVCClassifier',title=f"Accuracy: {clf.score(X_test, y_test)} - SVCClassifier\n{helper.get_special_name(folder_name='',prefix='')}")
