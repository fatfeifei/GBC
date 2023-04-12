import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,recall_score

file_dir = '../Expected results/Features_Radiomics.csv'
data = pd.read_csv(file_dir,index_col=0)

x = data.iloc[:,1:]
y = data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 420, stratify = y)


scaler = StandardScaler()
x_train_ss = scaler.fit_transform(x_train)
x_test_ss = scaler.transform(x_test)
x_train_ss = pd.DataFrame(x_train_ss,columns=x_train.columns,index=x_train.index)
x_test_ss = pd.DataFrame(x_test_ss,columns=x_test.columns,index=x_test.index)


selector_var = VarianceThreshold()
x_train_var = selector_var.fit_transform(x_train_ss)
x_test_var = selector_var.transform(x_test_ss)
x.columns[selector_var.get_support()]

X_train = pd.DataFrame(x_train_var, columns = x.columns[selector_var.get_support()],index=x_train.index)
X_test = pd.DataFrame(x_test_var, columns = x.columns[selector_var.get_support()],index=x_test.index)

model_all_features =DecisionTreeClassifier(random_state=0)
model_all_features.fit(X_train, y_train)
y_pred_test = model_all_features.predict_proba(X_test)[:, 1]

features = pd.Series(model_all_features.feature_importances_)
features.index = X_train.columns
features.sort_values(ascending=False, inplace=True)
features = list(features.index)


model_one_feature = SVC(kernel = 'rbf',class_weight = 'balanced',random_state = 0,probability=True)
model_one_feature.fit(X_train[features[0]].to_frame(), y_train)
result_test = model_one_feature.predict(X_test[features[0]].to_frame())
recall_score_first = recall_score(y_test,result_test)
print(recall_score_first)

print('doing recursive feature addition')
features_to_keep = [features[0]]
count = 1
for feature in features[1:]:
    print()
    print('testing feature: ', feature, ' which is feature ', count,
          ' out of ', len(features))
    count = count + 1
    model_int = SVC(kernel='rbf', class_weight='balanced', random_state=0, probability=True)

    model_int.fit(
        X_train[features_to_keep + [feature]], y_train)

    y_pred_test = model_int.predict(X_test[features_to_keep + [feature]])

    recall_score_int = recall_score(y_test, y_pred_test)
    print('New Test ROC AUC={}'.format((recall_score_int)))
    print('All features Test ROC AUC={}'.format((recall_score_int)))

    diff_auc = recall_score_int - recall_score_first

    if diff_auc >= 0.02:
        print('Increase in ROC AUC={}'.format(diff_auc))
        print('keep: ', feature)
        print

        recall_score_first = recall_score_int
        features_to_keep.append(feature)
    else:
        print('Increase in ROC AUC={}'.format(diff_auc))
        print('remove: ', feature)
        print
print('DONE!!')
print('total features to keep: ', len(features_to_keep))

print('Features kept:',features_to_keep)

final_model= SVC(kernel = 'rbf', class_weight = 'balanced',random_state = 0,probability=True)
final_model.fit(X_train[features_to_keep], y_train)

print('cross validation accuracy in the training dataset:', cross_val_score(final_model,X_train,y_train,cv=5).mean())
print('cross validation accuracy in the validation dataset:',cross_val_score(final_model,X_test,y_test,cv=5).mean())

auc_train = roc_auc_score(y_train, final_model.decision_function(X_train[features_to_keep]))
auc_test = roc_auc_score(y_test, final_model.decision_function(X_test[features_to_keep]))
print('AUC in the training datatset:', auc_train)
print('AUC in the validation datatset:', auc_test)

train_score = final_model.score(X_train[features_to_keep], y_train)
test_score = final_model.score(X_test[features_to_keep], y_test)
print('Accuracy in the training dataset:', train_score)
print('Accuracy in the validation dataset:',test_score)

result_train = final_model.predict(X_train[features_to_keep])
recall_train = recall_score(y_train,result_train)
result_test = final_model.predict(X_test[features_to_keep])
recall_test = recall_score(y_test,result_test)
print('Sensitivity in the training dataset:', recall_train)
print('Sensitivity in the validation dataset:', recall_test)