from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

from SVM.process_sample import load_n_process
from SVM.svm_setup import (svm_linear_fit, svm_predict,
                       train_and_save_svm_classifier)
from SVM.train_data_process import load_n_interpolate_train_data
from SVM.trtt_splitting import do_global_split

# TRAINING PHASE
dataset = load_n_interpolate_train_data() # is okay
# split the data
X_train, X_test, y_train, y_test, files_test = do_global_split(dataset)
# train the data - only need to this once
# save ytest filestest and comment code above
train_and_save_svm_classifier(X_train,y_train)
# single -> multi = false, many -> multi = true
multi = True
if multi == False:
    filename = f"training_data/{files_test[0]}.csv"
    # a brand new sample
    sample = load_n_process(filename)
    flat_sample = sample.flatten()
    ypred = svm_predict(flat_sample)
    print(f"True digit: {files_test[0]}")
    print(f"Predicted digit: {ypred}")
else:
    # Multisample prediction _ works but focus on a single sample
    ypred = []
    for i in range(len(X_test)):
        sample = X_test[i]
        pred = svm_predict(sample)
        ypred.append(pred)
    # check the stats
    accuracy = accuracy_score(y_test, ypred)
    print(accuracy)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, ypred)
    print(cm)
    print("\n")
    print("Classification Report:")
    print(classification_report(y_test, ypred))



