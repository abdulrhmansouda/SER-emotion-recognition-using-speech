# from sklearn.svm import SVC
# from X_Y import *
# import pickle
# import helper
# import time
# from sklearn.model_selection import train_test_split
# # this doesn't work with f0

# def confusion_matrix(y_test, y_prediction, title='', normalize=None):
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import ConfusionMatrixDisplay

#     class_names = para.emotions

#     disp = ConfusionMatrixDisplay.from_predictions(
#         y_true=y_test,
#         y_pred=y_prediction,
#         display_labels=class_names,
#         cmap=plt.cm.Blues,
#         normalize=normalize,
#     )

#     disp.ax_.set_title(title)


#     fig1 = plt.gcf()
#     plt.show()
#     plt.draw()
#     fig1.savefig(helper.get_special_name(folder_name='statistics\\SVC', prefix=str(
#         time.time())+'_SVC', extension='.png'), bbox_inches='tight')


# def get_classifier():
#     feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
#     X = feature_emotion_X_Y_array['X']
#     y = feature_emotion_X_Y_array['y']

#     # print(X)
#     # exit()
#     # clf = SVC(C=0.001, gamma=0.001, kernel='poly')
#     # clf = SVC(C=3, gamma=0.01, kernel='rbf',max_iter=100000)
#     import svm
#     return svm.train(X,y,3,'a')

#         #clf.fit(X=X, y=y)
#         #pickle.dump(clf, open(helper.get_special_name('pickled', 'SVC', '.pickle'), "wb"))
#     #return clf


# if __name__ == '__main__':

#     clf = get_classifier()
#     feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
#     X = feature_emotion_X_Y_array['X']
#     y = feature_emotion_X_Y_array['y']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = para.test_size, random_state = 0)


#     # exit()
#     y_prediction = clf.predict(X_test)
#     confusion_matrix(y_test=y_test, y_prediction=y_prediction,title=f"Accuracy: {clf.score(X_test, y_test)} - SVC\n{helper.get_special_name(folder_name='',prefix='')}")