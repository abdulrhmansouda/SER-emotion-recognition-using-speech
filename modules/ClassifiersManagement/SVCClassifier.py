from sklearn.svm import SVC
from X_Y import *
import pickle
import helper
import time
# this doesn't work with f0

def confusion_matrix(y_test, y_prediction, title='', normalize=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    class_names = para.emotions

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_prediction,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )

    disp.ax_.set_title(title)


    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(helper.get_special_name(folder_name='statistics\\SVC', prefix=str(
        time.time())+'_SVC', extension='.png'), bbox_inches='tight')


def get_classifier():
    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'SVC', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = get_feature_emotion_X_Y_array_for('train')
        X = feature_emotion_X_Y_array['X'].to_numpy()[:, 1:]
        # y = feature_emotion_X_Y_array['Y'].to_numpy()[:, 1:]
        y = np.argmax(feature_emotion_X_Y_array['Y'].to_numpy()[:,1:], axis=1)

        # clf = SVC(C=0.001, gamma=0.001, kernel='poly')
        clf = SVC(C=3, gamma=0.01, kernel='rbf',max_iter=100000)

        clf.fit(X=X, y=y)
        pickle.dump(clf, open(helper.get_special_name('pickled', 'SVC', '.pickle'), "wb"))
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
                     title=f"Accuracy: {clf.score(X_test, y_test)} - SVC\n{helper.get_special_name(folder_name='',prefix='')}")
    

"""
# from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix 
from X_Y import *

# hyper_parameter = {
#         'hidden_layer_sizes': [(200,), (300,), (400,), (128, 128), (256, 256)],
#         'alpha': [0.001, 0.005, 0.01],
#         'batch_size': [128, 256, 512, 1024],
#         'learning_rate': ['constant', 'adaptive'],
#         'max_iter': [200, 300, 400, 500]
#     }

feature_emotion_X_Y_array = get_feature_emotion_X_Y_array_for('train')
X = feature_emotion_X_Y_array['X'].to_numpy()[:,1:]
y = np.argmax(feature_emotion_X_Y_array['Y'].to_numpy()[:,1:], axis=1)

clf = SVC(C=0.001, gamma=0.001, kernel='poly')

# print(y)
# exit()

clf.fit(X=X, y=y)
# print('hello')

# exit()
feature_emotion_X_Y_array_for_test = get_feature_emotion_X_Y_array_for('test')
X_test = feature_emotion_X_Y_array_for_test['X'].to_numpy()[:,1:]
y_test = np.argmax(feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:,1:], axis=1)
# exit()
predicted_array_Y = clf.predict(X_test)

# print(np.argmax(predicted_array_Y, axis=1))
print(clf.score(X_test, y_test))

conf_matrix = confusion_matrix(y_test,predicted_array_Y)
print(conf_matrix)

"""