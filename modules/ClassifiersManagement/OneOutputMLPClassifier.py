from sklearn.neural_network import MLPClassifier
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
            'pickled', 'OneOutputMLPClassifier', '.pickle'), "rb"))
    except:
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


        clf = MLPClassifier(max_iter=500, learning_rate='adaptive',
                            hidden_layer_sizes=(200,), batch_size='auto', alpha=0.01)

        clf.fit(X=X_train, y=y_train)
        pickle.dump(clf, open(helper.get_special_name(
            'pickled', 'OneOutputMLPClassifier', '.pickle'), "wb"))
    # print('hello')
    return clf


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
    classifier = get_classifier()
    return helper.predict(path,classifier)

# predict('hi')

if __name__ == '__main__':

    clf = get_classifier()

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

    y_prediction = clf.predict(X_test)
    helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='OneOutputMLPClassifier',title=f"Accuracy: {clf.score(X_test, y_test)} - OneOutputMLPClassifier\n{helper.get_special_name(folder_name='',prefix='')}")

"""


from sklearn.neural_network import MLPClassifier
from X_Y import *
import pickle
import helper
import time


def confusion_matrix(y_test, y_prediction, title='', normalize=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    class_names = para.emotions

    disp = ConfusionMatrixDisplay.from_predictions(
        # classifier,
        y_true=y_test,
        y_pred=y_prediction,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )

    disp.ax_.set_title(title)

    # print(title)
    # print(disp.confusion_matrix)

    fig1 = plt.gcf()
    plt.show()
    # plt.close()
    plt.draw()
    fig1.savefig(helper.get_special_name(folder_name='statistics\\OneOutputMLPClassifier', prefix=str(
        time.time())+'_OneOutputMLPClassifier', extension='.png'), bbox_inches='tight')


def get_classifier():
    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'OneOutputMLPClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = get_feature_emotion_X_Y_array_for('train')
        X = feature_emotion_X_Y_array['X'].to_numpy()[:, 1:]
        # y = feature_emotion_X_Y_array['Y'].to_numpy()[:, 1:]
        y = np.argmax(feature_emotion_X_Y_array['Y'].to_numpy()[:,1:], axis=1)


        clf = MLPClassifier(max_iter=500, learning_rate='adaptive',
                            hidden_layer_sizes=(200,), batch_size='auto', alpha=0.01)

        clf.fit(X=X, y=y)
        pickle.dump(clf, open(helper.get_special_name(
            'pickled', 'OneOutputMLPClassifier', '.pickle'), "wb"))
    return clf


if __name__ == '__main__':

    clf = get_classifier()

    feature_emotion_X_Y_array_for_test = get_feature_emotion_X_Y_array_for(
        'test')

    X_test = feature_emotion_X_Y_array_for_test['X'].to_numpy()[:, 1:]
    # y_test = feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:, 1:]
    y_test = np.argmax(feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:,1:], axis=1)


    # exit()
    predicted_array_Y = clf.predict(X_test)
    confusion_matrix(y_test=y_test, y_prediction=predicted_array_Y,
                     title=f"Accuracy: {clf.score(X_test, y_test)} - OneOutputMLPClassifier\n{helper.get_special_name(folder_name='',prefix='')}")

    # # print(np.argmax(predicted_array_Y, axis=1))
    # print(clf.score(X_test, y_test))

    # conf_matrix = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(predicted_array_Y, axis=1))
    # print(conf_matrix)

"""