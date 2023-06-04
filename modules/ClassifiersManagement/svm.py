import os

curr_path = os.getcwd()

result_path = os.path.join(curr_path,'emotional results')

# Kernel SVM
def train(X,y,c_svm,feature):
    print("Kernel SVM")

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(C=c_svm,gamma=0.01,kernel = 'rbf', random_state =0,max_iter=100000)
    classifier.fit(X_train, y_train)

    # # Predicting the Test set results
    # y_pred = classifier.predict(X_test)

    # # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # #drawing confusion Matrix
    # labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

    # # classification report
    # from sklearn.metrics import classification_report
    # report = classification_report(y_test, y_pred, target_names=labels)
    # print(report)
    return classifier,sc