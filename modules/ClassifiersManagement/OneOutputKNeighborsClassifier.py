# from sklearn.neighbors import KNeighborsClassifier
# from X_Y import *
# import pickle
# import helper
# import time


# def confusion_matrix(y_test, y_prediction, title='', normalize=None):
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import ConfusionMatrixDisplay

#     class_names = para.emotions

#     disp = ConfusionMatrixDisplay.from_predictions(
#         # classifier,
#         y_true=y_test,
#         y_pred=y_prediction,
#         display_labels=class_names,
#         cmap=plt.cm.Blues,
#         normalize=normalize,
#     )

#     disp.ax_.set_title(title)

#     # print(title)
#     # print(disp.confusion_matrix)

#     fig1 = plt.gcf()
#     plt.show()
#     # plt.close()
#     plt.draw()
#     fig1.savefig(helper.get_special_name(folder_name='statistics\\OneOutputKNeighborsClassifier', prefix=str(
#         time.time())+'_OneOutputKNeighborsClassifier', extension='.png'), bbox_inches='tight')


# def get_classifier():
#     try:
#         clf = pickle.load(open(helper.get_special_name(
#             'pickled', 'OneOutputKNeighborsClassifier', '.pickle'), "rb"))
#     except:
#         feature_emotion_X_Y_array = get_feature_emotion_X_Y_array_for('train')
#         X = feature_emotion_X_Y_array['X'].to_numpy()[:, 1:]
#         # y = feature_emotion_X_Y_array['Y'].to_numpy()[:, 1:]
#         y = np.argmax(feature_emotion_X_Y_array['Y'].to_numpy()[:,1:], axis=1)

#         clf = KNeighborsClassifier(n_neighbors= 5, p= 1, weights= 'distance')

#         clf.fit(X=X, y=y)
#         pickle.dump(clf, open(helper.get_special_name(
#             'pickled', 'OneOutputKNeighborsClassifier', '.pickle'), "wb"))
#     return clf


# if __name__ == '__main__':

#     clf = get_classifier()

#     feature_emotion_X_Y_array_for_test = get_feature_emotion_X_Y_array_for(
#         'test')

#     X_test = feature_emotion_X_Y_array_for_test['X'].to_numpy()[:, 1:]
#     # y_test = feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:, 1:]
#     y_test = np.argmax(feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:,1:], axis=1)

#     # exit()
#     predicted_array_Y = clf.predict(X_test)
#     confusion_matrix(y_test=y_test, y_prediction=predicted_array_Y,
#                      title=f"Accuracy: {clf.score(X_test, y_test)} - OneOutputKNeighborsClassifier\n{helper.get_special_name(folder_name='',prefix='')}")