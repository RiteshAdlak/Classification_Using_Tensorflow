import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset has 800 rows
auction_data = pd.read_csv('x_test.csv')
auction_data.head()

# Performing Train Test Split using train_test_split

x_data = auction_data.drop('IsBadBuy',axis=1)
y_labels = auction_data['IsBadBuy']
# 30%(240 rows) reserved for test size 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=101)

# Creating Feature Columns
auction_data.columns

# Creating continous function for the continous values using numeric_column()

Auction = tf.feature_column.numeric_column("Auction")
VehicleAge = tf.feature_column.numeric_column("VehicleAge")
Make = tf.feature_column.numeric_column("Make")
Model = tf.feature_column.numeric_column("Model")
Color = tf.feature_column.numeric_column("Color")
WheelType = tf.feature_column.numeric_column("WheelType")
VehOdo = tf.feature_column.numeric_column("VehOdo")
Size = tf.feature_column.numeric_column("Size")
MMRAcquisitionAuctionAveragePrice = tf.feature_column.numeric_column("MMRAcquisitionAuctionAveragePrice")
MMRAcquisitionAuctionCleanPrice = tf.feature_column.numeric_column("MMRAcquisitionAuctionCleanPrice")
MMRAcquisitionRetailAveragePrice = tf.feature_column.numeric_column("MMRAcquisitionRetailAveragePrice")
MMRAcquisitonRetailCleanPrice = tf.feature_column.numeric_column("MMRAcquisitonRetailCleanPrice")
MMRCurrentAuctionAveragePrice = tf.feature_column.numeric_column("MMRCurrentAuctionAveragePrice")
MMRCurrentAuctionCleanPrice = tf.feature_column.numeric_column("MMRCurrentAuctionCleanPrice")
MMRCurrentRetailAveragePrice = tf.feature_column.numeric_column("MMRCurrentRetailAveragePrice")
MMRCurrentRetailCleanPrice = tf.feature_column.numeric_column("MMRCurrentRetailCleanPrice")
VNST = tf.feature_column.numeric_column("VNST")
VehBCost = tf.feature_column.numeric_column("VehBCost")
WarrantyCost = tf.feature_column.numeric_column("WarrantyCost")
BYRNO = tf.feature_column.numeric_column("BYRNO")

# Put all variables in single list 
feature_columns = [VehicleAge,WheelType,Size,VNST,Auction,Make,Model,Color,VehOdo,MMRAcquisitionAuctionAveragePrice,MMRAcquisitionAuctionCleanPrice,MMRAcquisitionRetailAveragePrice,MMRAcquisitonRetailCleanPrice,MMRCurrentAuctionAveragePrice,MMRCurrentAuctionCleanPrice,MMRCurrentRetailAveragePrice,MMRCurrentRetailCleanPrice,VehBCost,WarrantyCost,BYRNO]

# Creating Input Function
input_function = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)

# Creating model with tf.estimator
model = tf.estimator.LinearClassifier(input_function=feature_columns)

# Training model
model.train(input_fn=input_function,steps=5000)

# Evaluation : Creating prediction function
prediction_function = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
generator_of_prediction = model.predict(input_fn=prediction_function)
generator_of_prediction
predictions_list = list(generator_of_prediction)
prediction = [pred['class_ids'][0] for pred in predictions_list]
# Out of 240 rows(test data) counts for IsBadBuy : 0 and 1
prediction.count(1)			# 136
prediction.count(0)			# 104
prediction

# Classification Report

print(classification_report(y_test,prediction))

'''
				precision    recall  f1-score   support

          0       0.89      0.45      0.60       205
          1       0.18      0.69      0.28        35

avg / total       0.79      0.49      0.56       240
'''