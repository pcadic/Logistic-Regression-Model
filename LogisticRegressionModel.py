import pandas as pd
import numpy  as np
from sklearn import metrics

FOLDER = "C:\\TRASH\\datasets\\"
FILE   = 'amtrack_survey.csv'

#Read CSV file
print("Read CSV file")
df = pd.read_csv(FOLDER + FILE)

# Create separate copy to work on it easily
dfX = df.copy()
# Rename Seat Type into Class to avoid confusion with Seat type feature
dfX = dfX.rename(columns = {"Seat Type": "Class"})

# Definition of the target
y   = dfX['Satisfied']
del dfX['Satisfied']


# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Imputing function by mean value
def imputeNullValues(colName, df):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    del df[colName]     # Drop column with null values.
    return df

# replacing na values in Gender with Unknown
print("Fill null value for Gender feature")
dfX["Gender"].fillna("Unknown", inplace = True)

#Imputing values
print("Imputing Values for Delayed arrival")
dfX = imputeNullValues('Delayed arrival', dfX)
print("Imputing Values for Trip Distance")
dfX = imputeNullValues('Trip Distance', dfX)

#Getting dummies
print("Getting Dummies for 'Trip Type', 'Gender','Seat Type','Membership'")
dfX = pd.get_dummies(dfX, columns=['Trip Type','Gender','Class','Membership'])

#Bining
print("Bining categories for 'Age'")
dfX['Age bin']   = pd.cut(x=dfX['Age'], bins=[7, 27, 40, 51, 85])
dfX = pd.get_dummies(dfX, columns=['Age bin'])
#del dfX['Age']

print('Chi square calculation')
predictorVariables = list(dfX.keys())

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func=chi2, k='all' )
chiScores = test.fit(dfX, y)
np.set_printoptions(precision=3)
# Sort Chi square values
dfTemp = pd.DataFrame()
dfTemp['PredictorVariable'] = predictorVariables
dfTemp['chiScore'] = chiScores.scores_
sorted_df = dfTemp.sort_values(by='chiScore', ascending=False)
print('Sorted Chi square values')
print(sorted_df)


print("Logistic regression for models")
from sklearn.linear_model import LogisticRegression

#ModelA Perso by oservation of the numbers
# dfX = dfX[['imp_Trip Distance','Booking experience','Online experience', 'Membership_non-member',
#            'Class_premium' , 'Seat type', 'Checkin service', 'Gender_Male' , 'Quality Food',
#            'Trip Type_personal', 'Departure Convenience']]
#ModelB CHI
dfX = dfX[['imp_Trip Distance','imp_Delayed arrival','Delayed departure','Booking experience',
           'Online experience', 'Membership_non-member', 'Staff friendliness',
           'Boarding experience' , 'Class_premium' , 'Seat comfort' , 'Class_economy',
           'Seat type', 'Checkin service', 'Wifi', 'Luggage service', 'Cleanliness',
           'Gender_Male', 'Gender_Female', 'Membership_points membership', 'Age bin_(7, 27]',
           'Age bin_(40, 51]', 'Quality Food', 'Trip Type_personal' , 'Class_standard' ,
           'Trip Type_business', 'Age bin_(51, 85]', 'Age bin_(27, 40]' , 'Departure Convenience']]
#ModelC Forest Tree
# dfX = dfX[['Seat type', 'Booking experience', 'Gender_Male', 'Membership_non-member', 'Membership_points membership',
#             'Online experience', 'imp_Trip Distance', 'Boarding experience', 'Checkin service',
#            'Departure Convenience','Trip Type_personal', 'Trip Type_business', 'Gender_Female',
#             'Cleanliness', 'Seat comfort', 'Seat comfort', 'imp_Delayed arrival','Delayed departure',
#             'Wifi', 'Staff friendliness', 'Quality Food', 'Class_premium']]

print("Preparation of the cross-fold validation")
# enumerate splits - returns train and test arrays of indexes.
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

accuracyList  = []
precisionList = []
f1List        = []
recallList    = []
count = 0

# prepare cross validation with 10 folds.
kfold = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kfold.split(dfX):
    sc_x = StandardScaler()

    # Scale data.
    # Only fit on training data.
    X_train = dfX.loc[dfX.index.intersection(train_index), :]
    dfXScaled_Train = sc_x.fit_transform(X_train)

    # Transform test data.
    X_test = dfX.loc[dfX.index.intersection(test_index), :]
    dfXScaled_Test  = sc_x.transform(X_test)

    #  y does not need to be scaled since it is 0 or 1.
    y_train = y[train_index]
    y_test = y[test_index]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True,
                                       solver='liblinear')
    # Fit the model.
    logisticModel.fit(dfXScaled_Train, y_train)

    y_pred = logisticModel.predict(dfXScaled_Test)
    y_prob = logisticModel.predict_proba(dfXScaled_Test)

    # Show confusion matrix
    cm = pd.crosstab(y_test, y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])
    count += 1
    print("\n***K-fold: " + str(count))

    # Calculate accuracy and precision scores and add to the list.
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    accuracyList.append(accuracy)
    precisionList.append(precision)
    f1List.append(f1)
    recallList.append(recall)

    print("Accuracy:  ", round(accuracy,3))
    print("Precision: ", round(precision,3))
    print("Recall:    ", round(recall,3))
    print("F1:        ", round(f1,3))
    print("Confusion Matrix:")
    print(cm)

# Show averages of scores over multiple runs.
print("\nAccuracy and Standard Deviation For All Folds:")
print("*********************************************")
print("Average accuracy:  ", round(np.mean(accuracyList),3))
print("Accuracy std:      ", round(np.std(accuracyList),3))
print("Average precision: ", round(np.mean(precisionList),3))
print("Precision std:     ", round(np.std(precisionList),3))
print("Average recall:    ", round(np.mean(recallList),3))
print("Recall std:        ", round(np.std(recallList),3))
print("Average F1:        ", round(np.mean(f1List),3))
print("F1 std:            ", round(np.std(f1List),3))

print("END'")

