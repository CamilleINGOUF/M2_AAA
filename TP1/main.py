import pandas
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score,confusion_matrix
from sklearn.metrics import classification_report

def infoCSV(csv,col = None):
  print("=====info======")
  print(csv.shape)
  print(csv.head())
  print(csv.info())
  print(csv.describe())
  if col != None:
    print(csv[col].value_counts())
  print("===============")


def split_csv_x_y(csv,y_name, f):
  train,test = train_test_split(csv,test_size=0.3333)

  if f != None:
    train = f(train)

  x_train = train
  y_train = x_train[y_name]

  x_test = test
  y_test = x_test[y_name]

  del(x_train[y_name])
  # del(x_train[todelete])

  del(x_test[y_name])
  # del(x_test[todelete])
    
  return x_train,y_train, x_test, y_test

# Reduce class 0 count to Class 1 count
def delete_class_0(data):
  class_0 = data[data["Class"] == 0]
  class_1 = data[data["Class"] == 1]
  class_0 = class_0[:len(class_1)]
  class_0.append(class_1)
  return class_0


# Reduce class 0 count to Class 1 count
def duplicate_class_1(data):
  class_0 = data[data["Class"] == 0]
  class_1 = data[data["Class"] == 1]
  class_1 = pandas.concat([class_1]*500, ignore_index=0)
  class_0.append(class_1)
  return data

def do_algos(csv, f = None):
  x_train,y_train,x_test,y_test = split_csv_x_y(csv,"Class", f)

  # ======= TREE =========
  classifierTree = tree.DecisionTreeClassifier(criterion='entropy')
  classifierTree.fit(x_train,y_train)

  y_pred_tree = classifierTree.predict(X=x_test)
  print("===== TREE =====")
  print("r2 {}".format(r2_score(y_pred_tree,y_test)))
  print("MAE {}".format(mean_absolute_error(y_pred_tree,y_test)))
  print("MSE {}".format(mean_squared_error(y_pred_tree,y_test)))
  # print("Score {}".format(classifierTree.score(X=x_test,y=y_test)))
  print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_tree)))
  print("================")
  # ======================



  # ======= RFC =========
  RFC = RandomForestClassifier(n_estimators=10)
  RFC.fit(x_train,y_train)

  y_pred_rfc = RFC.predict(X=x_test)
  print("===== RFC =====")
  print("r2 {}".format(r2_score(y_pred_rfc,y_test)))
  print("MAE {}".format(mean_absolute_error(y_pred_rfc,y_test)))
  print("MSE {}".format(mean_squared_error(y_pred_rfc,y_test)))
  # print("Score {}".format(RFC.score(X=x_test,y=y_test)))
  print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_rfc)))
  print("================")
  # ======================



  # ======= KNN =========
  KNN = KNeighborsClassifier(n_neighbors=5)
  KNN.fit(x_train,y_train)

  y_pred_knn = KNN.predict(X=x_test)
  print("===== KNN =====")
  print("r2 {}".format(r2_score(y_pred_knn,y_test)))
  print("MAE {}".format(mean_absolute_error(y_pred_knn,y_test)))
  print("MSE {}".format(mean_squared_error(y_pred_knn,y_test)))
  # print("Score {}".format(KNN.score(X=x_test,y=y_test)))
  print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_knn)))
  print("================")
  # ======================



  # ======= MLPC =========
  mlpc = MLPClassifier()
  mlpc.fit(x_train,y_train)

  y_pred_mlpc = mlpc.predict(X=x_test)
  print("===== MLPC =====")
  print("r2 {}".format(r2_score(y_pred_mlpc,y_test)))
  print("MAE {}".format(mean_absolute_error(y_pred_mlpc,y_test)))
  print("MSE {}".format(mean_squared_error(y_pred_mlpc,y_test)))
  # print("Score {}".format(mlpc.score(X=x_test,y=y_test)))
  print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_mlpc)))
  print("================")
  # ======================

  # ======== Classification Reports ==========
  print("===== Classification Reports =====")
  print("Tree {}".format(classification_report(y_test, y_pred_tree)))
  print("RFC {}".format(classification_report(y_test, y_pred_rfc)))
  print("KNN {}".format(classification_report(y_test, y_pred_knn)))
  print("MLPC {}".format(classification_report(y_test, y_pred_mlpc)))
  print("==================================")
  # ==========================================

def main():
  csv = pandas.read_csv("creditcard.csv") 
  # infoCSV(csv, "Class")
  # do_algos(csv)  
  # do_algos(csv, delete_class_0)
  do_algos(csv, duplicate_class_1)


    
if __name__ == "__main__":
  main()