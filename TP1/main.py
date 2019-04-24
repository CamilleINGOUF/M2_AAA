import pandas
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score,confusion_matrix
from sklearn.metrics import classification_report

def infoCSV(csv,col):
  print("=====info======")
  print(csv.shape)
  print(csv.head())
  print(csv.info())
  print(csv.describe())
  if col != None:
    print(csv[col].value_counts())
  print("===============")


def split_csv_x_y(csv,y_name):
  train,test = train_test_split(csv,test_size=0.3333)

  x_train = train
  y_train = x_train[y_name]

  x_test = test
  y_test = x_test[y_name]

  del(x_train[y_name])
  # del(x_train[todelete])

  del(x_test[y_name])
  # del(x_test[todelete])
    
  return x_train,y_train, x_test, y_test

def main():
  csv = pandas.read_csv("creditcard.csv") 
  # infoCSV(csv, "Class")  
  x_train,y_train,x_test,y_test = split_csv_x_y(csv,"Class")

  # ======= TREE =========
  classifierTree = tree.DecisionTreeClassifier(criterion='entropy')
  classifierTree.fit(x_train,y_train)

  y_pred_tree = classifierTree.predict(X=x_test)
  print("===== TREE =====")
  print("r2 {}".format(r2_score(y_pred_tree,y_test)))
  print("MAE {}".format(mean_absolute_error(y_pred_tree,y_test)))
  print("MSE {}".format(mean_squared_error(y_pred_tree,y_test)))
  print("Score {}".format(classifierTree.score(X=x_test,y=y_test)))
  print("Confusion Matrix {}".format(confusion_matrix(y_test,y_pred_tree)))
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
  print("Score {}".format(RFC.score(X=x_test,y=y_test)))
  print("Confusion Matrix {}".format(confusion_matrix(y_test,y_pred_rfc)))
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
  print("Score {}".format(RFC.score(X=x_test,y=y_test)))
  print("Confusion Matrix {}".format(confusion_matrix(y_test,y_pred_knn)))
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
  print("Score {}".format(RFC.score(X=x_test,y=y_test)))
  print("Confusion Matrix {}".format(confusion_matrix(y_test,y_pred_mlpc)))
  print("================")
  # ======================

  # ======== Classification Reports ==========
  
  # ==========================================


    
if __name__ == "__main__":
  main()