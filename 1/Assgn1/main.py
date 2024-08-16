import math
from random import sample

import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np



### spliting datasets
def splitting_data(df_insurance):
    training_data = df_insurance.sample(frac=0.8,random_state=45)
    testing_data = df_insurance.drop(training_data.index)
    return training_data, testing_data


def outliers(df_insurance, columns):
    limit =  [0]*(len(columns))
    for i in range(1, len(columns)):
        limit[i] = df_insurance[columns[i]].mean() + (3 * df_insurance[columns[i]].std())
    for j in range(1, len(columns)):
        index = df_insurance[(df_insurance[columns[i]]) > limit[j]].index
    if len(index) != 0:
        #print("OK")
        df_insurance.drop(index, inplace=True)
    return



def if_value_null(df_insurance, columns):

    for i in range(1,len(columns)-1):
        for j in range(0,df_insurance.shape[0]):
            if df_insurance.loc[j][columns[i]] == None:
                df_insurance.loc[j][columns[i]] = df_insurance[columns[i]].mean()
                #print("OK")

def k_fold_train(training,laplace):
    num_folds = 10
    subset_size = len(training) / num_folds
    accuracy = [0]*num_folds
    training = training.reindex(np.random.permutation(training.index))
    #print(training)
    training = training.reset_index(drop=True)
    #print(training)
    fold = [0]*10
    for i in range(num_folds):
        fold[i] = training.loc[i*subset_size:((i+1)*subset_size)-1]
        #testing_this_round = training[int(i * subset_size):][:int(subset_size)]
        #training_this_round = training[:int(i * subset_size)] + training[int((i + 1) * subset_size):]

    train_val = [0]*10
    train_val[0] = training.drop(fold[0].index)
    train_val[1] = training.drop(fold[1].index)
    train_val[2] = training.drop(fold[2].index)
    train_val[3] = training.drop(fold[3].index)
    train_val[4] = training.drop(fold[4].index)
    train_val[5] = training.drop(fold[5].index)
    train_val[6] = training.drop(fold[6].index)
    train_val[7] = training.drop(fold[7].index)
    train_val[8] = training.drop(fold[8].index)
    train_val[9] = training.drop(fold[9].index)

    for i in range(num_folds):
        y_train = train_val[i]["Response"]
        x_train = train_val[i].drop("Response", axis=1)

        y_test = fold[i]["Response"]
        x_test = fold[i].drop("Response", axis=1)

        means = train_val[i].groupby(["Response"]).mean()                                               # Find mean of each class

        var = train_val[i].groupby(["Response"]).var()                                                  # Find variance of each class
        prior = ((train_val[i].groupby("Response").count() + laplace) / (len(train_val[i]) + laplace)).iloc[:, 1]                # Find prior probability of each class    len(train_val[i].columns)
        classes = np.unique(train_val[i]["Response"].tolist())                                          # Storing all possible classes

        pred_x_test = Predict(x_test, means, var, prior, classes, x_test)

        accuracy[i] = round(100*Accuracy(y_test, pred_x_test), 5)
        #print(round(100*Accuracy(y_train, pred_x_train), 5))

    max_value = max(accuracy)
    index = accuracy.index(max_value)

    return train_val[index]


def Normal(x, mu, var):                                       # Function to return pdf of Normal(mu, var) evaluated at x

    stdev = np.sqrt(var)
    pdf = (np.e ** (-0.5 * ((x - mu) / stdev) ** 2)) / (stdev * np.sqrt(2 * np.pi))

    return pdf


def Predict(X, means, var, prior, classes, x_train):
    Predictions = []
    #print(x_train)
    #print(X.loc[3])
    for i in X.index:                                           # Loop through each instances

        ClassLikelihood = []
        instance = X.loc[i]
        #print(classes)
        for cls in classes:                                     # Loop through each class

            FeatureLikelihoods = []
            FeatureLikelihoods.append(np.log(prior[cls]))       # Append log prior of class 'cls'

            for col in x_train.columns:                         # Loop through each feature

                data = instance[col]

                mean = means[col].loc[cls]                      # Find the mean of column 'col' that are in class 'cls'
                variance = var[col].loc[cls]                    # Find the variance of column 'col' that are in class 'cls'
                #print("OK")
                Likelihood = Normal(data, mean, variance)

                if Likelihood != 0:
                    Likelihood = np.log(Likelihood)             # Find the log-likelihood evaluated at x
                else:
                    Likelihood = 1 / len(train)

                FeatureLikelihoods.append(Likelihood)

            TotalLikelihood = sum(FeatureLikelihoods)           # Calculate posterior
            ClassLikelihood.append(TotalLikelihood)

        MaxIndex = ClassLikelihood.index(max(ClassLikelihood))  # Find the largest posterior position
        Prediction = classes[MaxIndex]
        Predictions.append(Prediction)
    #print(FeatureLikelihoods)
    return Predictions


def Accuracy(y, prediction):
    y = list(y)
    prediction = list(prediction)
    score = 0

    for i, j in zip(y, prediction):
        if i == j:
            score += 1

    return score / len(y)




if __name__ == "__main__":
    #### reading data from CSV
    df_insurance = pd.read_csv(r'Dataset_C.csv')
    df_orig = df_insurance

    le = LabelEncoder()
    df_insurance['Gender'] = le.fit_transform(df_insurance['Gender'])
    df_insurance['Vehicle_Age'] = le.fit_transform(df_insurance['Vehicle_Age'])
    df_insurance['Vehicle_Damage'] = le.fit_transform(df_insurance['Vehicle_Damage'])

    #print(df_insurance.loc[0]['Gender'])

    #if_value_null(df_insurance, df_insurance.columns)
    outliers(df_insurance, df_insurance.columns)

    #var_data = df_insurance.groupby(["Response"]).var()
    #var_data.to_csv(r'varorgpres.csv')

    df_insurance = df_insurance.drop("id",axis=1)
    #df_insurance = df_insurance.drop("Driving_License",axis=1)
    #df_insurance = df_insurance.drop("Previously_Insured",axis=1)
    #df_insurance = df_insurance.drop("Vehicle_Damage",axis=1)

    print(df_insurance.head(1))

    train, test = splitting_data(df_insurance)

    laplace = 1

    k_fold_train_ds = k_fold_train(train,laplace)

    k_fold_train_y = k_fold_train_ds["Response"]
    k_fold_train_x = k_fold_train_ds.drop("Response",axis=1)

    #print(k_fold_train_ds.groupby("Response").count())

    df_means = k_fold_train_ds.groupby(["Response"]).mean()                                                          # Find mean of each class
    df_var = k_fold_train_ds.groupby(["Response"]).var()                                                             # Find variance of each class
    df_prior = ((k_fold_train_ds.groupby("Response").count() + laplace) / ((len(k_fold_train_ds))+ laplace )).iloc[:, 1]         # Find prior probability of each class      len(k_fold_train_ds.columns)
    df_classes = np.unique(k_fold_train_ds["Response"].tolist())                                                     # Storing all possible classes

    #df_means = k_fold_train_ds.groupby(["Response"]).mean()                                                         # Find mean of each class
    #df_var = k_fold_train_ds.groupby(["Response"]).var()                                                            # Find variance of each class
    #df_prior = (k_fold_train_ds.groupby("Response").count() / (len(k_fold_train_ds))).iloc[:,1]                     # Find prior probability of each class
    #df_classes = np.unique(k_fold_train_ds["Response"].tolist())                                                    # Storing all possible classes



    y_test = test["Response"]
    x_test = test.drop("Response",axis=1)

    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


    #PredictTest = Predict(x_test, df_means, df_var, df_prior, df_classes, x_test)

    #print("Accuracy:", round(100 * Accuracy(y_test, PredictTest), 5) )

