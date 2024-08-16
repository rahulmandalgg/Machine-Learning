import math
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


#function to split data into training and test set
def splitting_data(df_data):
    training_data = df_data.sample(frac=0.8,random_state=18)
    testing_data = df_data.drop(training_data.index)
    return training_data, testing_data


#function to handle missing data
def if_value_null(df_data):

    for i in range(len(df_data.columns)):

        for j in range(len(df_data[i])):
            if np.isnan(df_data[i][j]) == True :
                df_data[i][j] = math.ceil(df_data[i].mean())

    return df_data

#function for standard scalar normalization
def standard_scalar_normalize(data):
    #print(data)
    for i in range(1, len(data.columns) ):

        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - data[i].mean()) / data[i].std()

    return data

def forward_sel(X,X_train,y_train,X_test,y_test,mlp_model):
    cols = X_train.columns
    best_cols = []
    best_acc = 0
    #print(cols)
    for i in range(len(cols)):
        best_col = ''
        for col in cols:
            if col not in best_cols:
                mlp_model.fit(X_train[best_cols + [col]], y_train)
                pred = mlp_model.predict(X_test[best_cols + [col]])
                acc = accuracy_score(pred, y_test)
                #print(acc)
                if acc >= best_acc:
                    best_acc = acc
                    best_col = col
        if best_col != '':
            best_cols.append(best_col)
            #print(best_acc)
    print('Best features are :')
    return best_cols

if __name__ == "__main__":
    #### reading data file
    df_orig = pd.read_csv('lung-cancer.data', na_values= "?", header= None)
    df_data = df_orig


    df_data = if_value_null(df_orig)
    df_data = standard_scalar_normalize(df_data)
    df_target = df_data[0]
    df_data = df_data.drop(0,axis=1)

    #print(df_target)

    #splitting data into 80:20 training and test set
    train_X, test_X = splitting_data(df_data)
    train_y, test_y = splitting_data(df_target)

    #applying svm with different kernels

    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(train_X,train_y)

    poly = svm.SVC(kernel='poly', degree=2, C=1, decision_function_shape='ovo').fit(train_X,train_y)

    radial = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(train_X,train_y)


    #prediction and finding accuracy for all svm kernels

    pred_lin = linear.predict(test_X)
    accuracy_lin = accuracy_score(test_y,pred_lin)
    print("Accuracy for SVM Classifier using linear kernel is: ")
    print(accuracy_lin*100)

    pred_poly = poly.predict(test_X)
    accuracy_quad = accuracy_score(test_y,pred_poly)
    print("Accuracy for SVM Classifier using quadratic kernel is: ")
    print(accuracy_quad*100)

    pred_radial = radial.predict(test_X)
    accuracy_radial = accuracy_score(test_y,pred_radial)
    print("Accuracy for SVM Classifier using radial kernel is: ")
    print(accuracy_radial*100)

    #MLP classifier with different hidden layers
    mlp_classf1 = MLPClassifier(solver='sgd', hidden_layer_sizes=(16), batch_size=32, learning_rate_init=0.001, random_state=0).fit(train_X,train_y)
    pred_mlp_test1 = mlp_classf1.predict(test_X)

    mlp_test_score1 = accuracy_score(pred_mlp_test1,test_y)

    print("Accuracy for MLP Classifier using Stochastic Descent Gradient with 1 hidden layer with 16 nodes is: ")
    print(mlp_test_score1*100)

    mlp_classf2 = MLPClassifier(solver='sgd', hidden_layer_sizes=(256,16), batch_size=32, learning_rate_init=0.001, random_state=0).fit(train_X, train_y)
    pred_mlp_test2 = mlp_classf2.predict(test_X)

    mlp_test_score2 = accuracy_score(pred_mlp_test2, test_y)

    print("Accuracy for MLP Classifier using Stochastic Descent Gradient with 2 hidden layers with 256 and 16 nodes is: ")
    print(mlp_test_score2*100)

    if mlp_test_score1 > mlp_test_score2:
        modeltosel = [16]
    else: modeltosel = [256,16]


    learn_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    accfscore = [0,0,0,0,0]

    #varying learning rate in MLP classfier
    for i in range(0,len(learn_rate)):
        classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=modeltosel, batch_size=32, learning_rate_init=learn_rate[i],random_state=0).fit(train_X, train_y)
        predfinal = classifier.predict(test_X)
        accfscore[i] = accuracy_score(predfinal,test_y) * 100


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(learn_rate, accfscore, 'r*', linestyle="--")


    for xy in zip(learn_rate, accfscore):
        ax.annotate('(%.6f, %.2f)' % xy, xy=xy)

    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy ")

    #plotting Learning Rate vs accuracy
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.show()


    #forward selection method
    bstFtSel_model = MLPClassifier(solver='sgd', hidden_layer_sizes=modeltosel, batch_size=32,learning_rate_init=0.001, random_state=0)
    features = forward_sel(df_orig,train_X,train_y,test_X,test_y,bstFtSel_model)
    features.sort()
    print(features)
    string_feat = "Final Set of Features with Highest Accuracy Score are: "
    f = open('Output.txt', 'w')
    f.write(str(string_feat))
    f.write("\n \t\t ")
    f.write(str(features))
    f.close()


    #applying ensemble learning
    models = []
    modl1 = poly
    modl2 = radial
    modl3 = MLPClassifier(solver='sgd', hidden_layer_sizes=modeltosel, batch_size=32, learning_rate_init=0.001, random_state=0).fit(train_X,train_y)
    models = [("Quad",modl1), ("Radial",modl2), ("MLP",modl3)]

    modl3_pred = modl3.predict(test_X)

    ensemble_pred = [[0]*3]*len(test_y)
    for i in range(0,len(test_y)):
        ensemble_pred[i] = [pred_radial[i], pred_poly[i],modl3_pred[i]]
    array_ens = np.array(ensemble_pred)

    max_voting_pred = [0]*len(test_y)
    for i in range(0,len(test_y)):
        max_voting_pred[i] = np.bincount(array_ens[i]).argmax()

    results = accuracy_score(test_y,max_voting_pred)

    print("Final Accuracy after ensemble learning with max voting method is: ")
    print(100*results)






