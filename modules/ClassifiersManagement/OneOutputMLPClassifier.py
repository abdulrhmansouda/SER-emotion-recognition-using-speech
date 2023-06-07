# from sklearn.neural_network import MLPClassifier
# import pickle
# import helper
# from sklearn.model_selection import train_test_split
# import sys
# import os
# sys.path.insert(0, os.getcwd())
# import parameters as para
# sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
# from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array




# def get_classifier():
#     try:
#         clf = pickle.load(open(helper.get_special_name(
#             'pickled', 'OneOutputMLPClassifier', '.pickle'), "rb"))
#     except:
#         feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
#         X = feature_emotion_X_Y_array['X']
#         y = feature_emotion_X_Y_array['y']
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=para.test_size, random_state=0)
#         # Feature Scaling
#         # from sklearn.preprocessing import StandardScaler
#         # sc = StandardScaler()
#         # X_train = sc.fit_transform(X_train)
#         # X_test = sc.transform(X_test)

#         clf = MLPClassifier(max_iter=500, learning_rate='adaptive',
#                             hidden_layer_sizes=(200,), batch_size='auto', alpha=0.01)

#         clf.fit(X=X_train, y=y_train)
#         pickle.dump(clf, open(helper.get_special_name(
#             'pickled', 'OneOutputMLPClassifier', '.pickle'), "wb"))
#     # print('hello')
#     return clf


# def get_classifier_through_randomized_search_cv():
#     from sklearn.neural_network import MLPClassifier
#     from sklearn.model_selection import RandomizedSearchCV

#     try:
#         random_search = pickle.load(open(helper.get_special_name(
#             'pickled', 'RandomizeSearch_OneOutputMLPClassifier', '.pickle'), "rb"))
#     except:
#         feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
#         X = feature_emotion_X_Y_array['X']
#         y = feature_emotion_X_Y_array['y']
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=para.test_size, random_state=0)

#         # Define the parameter grid to search over
#         param_dist = {
#             'max_iter': [500, 1000],
#             'learning_rate': ['constant', 'adaptive'],
#             'batch_size': ['auto'],
#             'alpha': [0.001, 0.005, 0.01],
#             'hidden_layer_sizes': [(64,), (128,), (256,), (512,), (64, 64), (128, 128), (256, 256), (512, 256), (512, 512)],
#             'activation': ['relu', 'tanh'],
#             'solver': ['adam', 'sgd']

#             # 'solver': ['adam'],
#             # 'max_iter': [500],
#             # 'learning_rate': ['adaptive'],
#             # 'hidden_layer_sizes': [(100, 100), ],
#             # 'batch_size': ['auto'],
#             # 'alpha': [0.01],
#             # 'activation': ['relu'],

#         }

#         # Create an MLPClassifier object
#         clf = MLPClassifier()

#         # Create a RandomizeSearchCV object
#         random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

#         # Fit the grid search object to the data
#         random_search.fit(X=X_train, y=y_train)
#         pickle.dump(random_search, open(helper.get_special_name(
#             'pickled', 'RandomizeSearch_OneOutputMLPClassifier', '.pickle'), "wb"))
#     # Print the best parameters and best score
#     print("Best parameters: ", random_search.best_params_)
#     print("Best score: ", random_search.best_score_)

#     return random_search


# def predict(path):
#     classifier = get_classifier()
#     return helper.predict(path, classifier)


# if __name__ == '__main__':
#     # clf = get_classifier()
#     clf = get_classifier_through_randomized_search_cv()

#     feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
#     X = feature_emotion_X_Y_array['X']
#     y = feature_emotion_X_Y_array['y']
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=para.test_size, random_state=0)

#     y_prediction = clf.predict(X_test)
#     helper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_OneOutputMLPClassifier',
#                             title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_OneOutputMLPClassifier\n{helper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")
