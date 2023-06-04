import time
import os
import sys
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeatuersManagement")
from modules.FeatuersManagement.main import extract_feature_emotion_X_y_array
from modules.FeatuersManagement.Feature import Feature


def get_special_name(folder_name=para.feature_emotion_X_Y_array_folder_path_name, prefix='', extension=''):
    name = prefix
    for dataset in para.datasets:
        name = name + \
            f"_{dataset}"
    for feature in para.features:
        name = name + \
            f"_{feature}"
    for emotion in para.emotions:
        name = name + \
            f"_{emotion}"

    name = os.path.join(folder_name, f"{name}{extension}")
    return name


def confusion_matrix(y_test, y_prediction, classifier_name, title='', normalize=None):
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
    fig1.savefig(get_special_name(folder_name=f'statistics\\{classifier_name}', prefix=str(
        time.time())+f'_{classifier_name}', extension='.png'), bbox_inches='tight')

def predict(path,classifier,sc):
    # from sklearn.model_selection import train_test_split
    # feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    # X = feature_emotion_X_Y_array['X']
    # y = feature_emotion_X_Y_array['y']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = para.test_size, random_state = 0)
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    feature = sc.transform(Feature(path).extract_feature().reshape(1,-1))
    # feature = Feature(path).extract_feature().reshape(1,-1)

    return para.emotions[int(classifier.predict(feature)[0])]
    # classifier = get_classifier()