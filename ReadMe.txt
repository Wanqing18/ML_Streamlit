
This web aims to predict the churn rate (https://www.investopedia.com/terms/c/churnrate.asp) of a customer using XGBoosting with cutomized hyperparameters. 


#Data
Data are provided by TD in WAFn-UseC-Telco-Customer-Churn.csv file, it contains information relate to Churn Rate Prediction. This data is only for data-challeng and learning purpose.

#Before Start:
1.Though clicking 'Upload','Profiling','Modeling' and 'Download' on the sidebar, user could complete the end-to-end training process including data preprocessing, Hyperparameter tuning, model fitting, and performance evaluation
2.But without uploading data, function in the 'Profiling','Modeling' and 'Download' Session would not be avaliable.
3.In order to access the final result, User should follow the order from 'Uploading' to 'Download Session' 
4.But 'Profiling' Session is independent from 'Modeling' and 'Download' Sessions, user could skip the 'Profiling' to Modeling

#Upload Session
1.User could drag or upload csv file from local folder(only 1 file could be processed each time)
2.After file get uploaded, user could view the dataset.
3.Once see 'File has been uploaded', user could move to next step

#Profiling Session
1.By pressing 'start profiling', an automated Exploratory Data Analysis wil be performed. 
2.Users could have a quick view on basic information of each varaible, such as distribution, if there is missing values, shape and type,etc

# Modeling Session
1. user should first choose the target variable that used for prediction,in this case 'Churn' will be selected
2.By pressing "Start Processing", data cleaning and preprocessing would be completed, Features(Independent variables) and Labels(Dependet Variable) could be viewed, once processing is finished
3.User should then set up 'spliting ratio of training data' to split training and test dataset. Training data is for model training, Testing data is used for evaluating model's performance
4. Then user could select their prefered hyoerparameters.
    Learning Rate: how much to change the model in response to the estimated error each time the model weights are updated. smaller learning rate will increase complexity and more likely to overfit
    #Keep Increasing value below will make the model more complex and more likely to overfit.
    Max Depth:Maximum depth of a tree. 
    Numbers of Estimators:The number of trees (or rounds)
    Colsample_bytree:is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
5.By pressing 'Predict' model will start to train and generate result
6. After seeing 'prediction finished' user could select output that they want to see in the multiple select bar


#Download Session
1.After finishing predictions, user could download their prefered output by selecting different tags in multiple select bar.
         Curve and plot would be saved as png file;
         outcome and Metric report would be saved as csv file;
         model has been download as pkl file
            to upload and use the model for future prediction:
            #pickled_model = pickle.load(open('model.pkl', 'rb'))
            pickled_model.predict(X_test_OE)