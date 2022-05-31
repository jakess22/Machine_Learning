import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_score
from statistics import mean
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy
import keras

    
X = pd.read_csv('Dry_Beans_Dataset.csv')
y = X['Class']
del X['Class'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 20)

# Min-Max Normalization
sc_X = MinMaxScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.fit_transform(X_test)

print("Train shape (X, y):", X_train_scaled.shape, y_train.shape)
print("Test shape (X, y):", X_test_scaled.shape, y_test.shape)

# Build neural network
# citation (3) : scikit-learn.org
model = MLPClassifier(hidden_layer_sizes = (12, 3), activation = "logistic", solver = "sgd", learning_rate_init = 0.3, max_iter = 500)


model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Model score:", model.score(X_test_scaled, y_test))
print("Model MSE:", model.loss_)

# Precision, recall, and accuracy
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels = ['BAR', 'BOM', 'CA', 'DER', 'HOR', 'SEK', 'SIRA'])
cm_display.plot()

# k-Fold cross validation
# citation (2) : "Discussion_4"


acc_per_fold = []
loss_per_fold = []
fold_num = 1

kfold = KFold(n_splits = 10, shuffle = True)
for train, test in kfold.split(X, y):
    # Train on different folds
    model = MLPClassifier(hidden_layer_sizes = (12, 3), activation = "logistic", solver = "sgd", learning_rate_init = 0.3, max_iter = 500)
    hist = model.fit(X.iloc[train], y.iloc[train])
    scores = cross_val_score(model, X, y, cv = 10)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Next fold number
    fold_num = fold_num + 1
print("Accuracy per fold:")
print(acc_per_fold)
print("Average accuracy:", mean(acc_per_fold))
print("MSE (loss) per fold:")
print(loss_per_fold)
print("Average MSE per fold:", mean(loss_per_fold))

# Hyperparamter tuning


# citation (4) : "gridsearch_example"
# citation (5) : "Grid Search Optimization Algorithm in Python"

# README: do not run unless you have 2-4 hours to complete the tuning
def create_model(neurons = 12, learning_rate = 0.3):
    model = Sequential()
    model.add(Dense(neurons, activation = 'sigmoid'))
    model.add(Dense(neurons, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = SGD(learning_rate = learning_rate), metrics = ['accuracy'])
    return model

seed = 5
numpy.random.seed(seed)
model = KerasClassifier(build_fn = create_model)
neurons_list = [3, 6, 12]
learn_rate_list = [.01, .1, .3]
epochs_list = [1, 50, 500]

#parameters = dict(learning_rate = learn_rate_list, epochs = epochs_list, neurons = neurons_list)
#grid = GridSearchCV(estimator = model, cv = 3, param_grid = parameters)

#grid_result = grid.fit(X_train, y_train)
#print("Maximum accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


