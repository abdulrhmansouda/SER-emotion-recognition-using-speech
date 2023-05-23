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
        y_true=np.argmax(y_test, axis=1),
        y_pred=np.argmax(y_prediction, axis=1),
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
    fig1.savefig(helper.get_special_name(folder_name='statistics\\MLPClassifier', prefix=str(
        time.time())+'_MLPClassifier', extension='.png'), bbox_inches='tight')
    # plt.savefig(helper.get_special_name('pickled', str(time.time())+'_MLPClassifier','.png'),dpi=300)


def get_classifier():
    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'MLPClassifier', '.pickle'), "rb"))
    except:
        feature_emotion_X_Y_array = get_feature_emotion_X_Y_array_for('train')
        X = feature_emotion_X_Y_array['X'].to_numpy()[:, 1:]

        y = feature_emotion_X_Y_array['Y'].to_numpy()[:, 1:]

        # clf = MLPClassifier(max_iter=500, learning_rate='adaptive',hidden_layer_sizes=(200,), batch_size='auto', alpha=0.01)
        clf = MLPClassifier(solver= 'adam', max_iter= 500, learning_rate= 'adaptive', hidden_layer_sizes= (256, 256), batch_size= 'auto', alpha= 0.005, activation= 'tanh')

        clf.fit(X=X, y=y)
        pickle.dump(clf, open(helper.get_special_name(
            'pickled', 'MLPClassifier', '.pickle'), "wb"))
    return clf

def get_classifier_through_grid_search():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    # from sklearn.datasets import make_classification

    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'GridSearch_MLPClassifier', '.pickle'), "rb"))
        return clf
    except:

    # Generate a random dataset
        feature_emotion_X_Y_array = get_feature_emotion_X_Y_array_for('train')
        X = feature_emotion_X_Y_array['X'].to_numpy()[:, 1:]
        y = feature_emotion_X_Y_array['Y'].to_numpy()[:, 1:]
        # X, y = make_classification(n_features=3000)

        # Define the parameter grid to search over
        param_grid = {
            'max_iter': [500, 1000],
            'learning_rate': ['constant','adaptive'],
            'batch_size': ['auto'],
            'alpha': [0.001, 0.005, 0.01],
            'hidden_layer_sizes': [(50,), (100,), (256,), (512,), (50, 50), (100, 100), (256, 256), (512, 256), (512, 512)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
            # 'max_iter': [500],
            # 'learning_rate': ['constant'],
            # 'batch_size': ['auto'],
            # 'alpha': [0.001,],
            # 'hidden_layer_sizes': [(50,),],
            # 'activation': ['relu',],
            # 'solver': ['adam',]
        }

        # Create an MLPClassifier object
        mlp = MLPClassifier()

        # Create a GridSearchCV object
        grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=5,verbose=3)
        # grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=2,verbose=3)

        # Fit the grid search object to the data
        grid_search.fit(X, y)
        pickle.dump(grid_search, open(helper.get_special_name(
            'pickled', 'GridSearch_MLPClassifier', '.pickle'), "wb"))
        # Print the best parameters and best score
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)

def get_classifier_through_randomized_search_cv():
    from sklearn.neural_network import MLPClassifier
    # from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV

    # from sklearn.datasets import make_classification

    try:
        clf = pickle.load(open(helper.get_special_name(
            'pickled', 'RandomizeSearch_MLPClassifier', '.pickle'), "rb"))
        return clf
    except:

        # Generate a random dataset
        feature_emotion_X_Y_array = get_feature_emotion_X_Y_array_for('train')
        X = feature_emotion_X_Y_array['X'].to_numpy()[:, 1:]
        y = feature_emotion_X_Y_array['Y'].to_numpy()[:, 1:]
        # X, y = make_classification(n_features=3000)

        # Define the parameter grid to search over
        param_dist = {
            'max_iter': [500, 1000],
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': ['auto'],
            'alpha': [0.001, 0.005, 0.01],
            'hidden_layer_sizes': [(50,), (100,), (256,), (512,), (50, 50), (100, 100), (256, 256), (512, 256), (512, 512)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
            # 'max_iter': [500],
            # 'learning_rate': ['constant'],
            # 'batch_size': ['auto'],
            # 'alpha': [0.001,],
            # 'hidden_layer_sizes': [(50,),],
            # 'activation': ['relu',],
            # 'solver': ['adam',]
        }

        # Create an MLPClassifier object
        mlp = MLPClassifier()

        # Create a RandomizeSearchCV object
        random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

        # Fit the grid search object to the data
        random_search.fit(X, y)
        pickle.dump(random_search, open(helper.get_special_name(
            'pickled', 'RandomizeSearch_MLPClassifier', '.pickle'), "wb"))
        # Print the best parameters and best score
        print("Best parameters: ", random_search.best_params_)
        print("Best score: ", random_search.best_score_)


if __name__ == '__main__':

    # clf = get_classifier_through_grid_search()
    # clf = get_classifier_through_randomized_search_cv()
    clf = get_classifier()
    # exit()

    feature_emotion_X_Y_array_for_test = get_feature_emotion_X_Y_array_for('test')

    X_test = feature_emotion_X_Y_array_for_test['X'].to_numpy()[:, 1:]
    y_test = feature_emotion_X_Y_array_for_test['Y'].to_numpy()[:, 1:]

    # exit()
    predicted_array_Y = clf.predict(X_test)
    confusion_matrix(y_test=y_test, y_prediction=predicted_array_Y,
                     title=f"Accuracy: {clf.score(X_test, y_test)} - MLPClassifier\n{helper.get_special_name(folder_name='',prefix='')}")

    # # print(np.argmax(predicted_array_Y, axis=1))
    # print(clf.score(X_test, y_test))

    # conf_matrix = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(predicted_array_Y, axis=1))
    # print(conf_matrix)
