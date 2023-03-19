import pandas as pd
from utils import * 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
from scikitplot import metrics
import scikitplot as skplt
import io

class ChurnRateModel:
    def __init__(self):
        self.df = pd.DataFrame()
        self.target= None
        self.X_test_OE = None
        self.X_train_OE = pd.DataFrame()
        self.y_train = None
        self.y_test = None
        self.features = pd.DataFrame()
        self.labels= None
        self.xgbModel = None
        self.xgbClassTest = None
        self.xg_probs_test0 = None
        self.xg_probs_test = None
        self.importancePlot = None
        self.confusionPlot = None
        self.roc_curvePlot = None
        self.multiplot = None
        self.report = None
        self.saved_model = None
        self.confusionMatrixImage = io.BytesIO()
        self.roc_curvePlotImage = io.BytesIO()
        self.ImportanceFeaturesImage=io.BytesIO()
        self.Multiple_ROCPlotImage=io.BytesIO()
        self.figsize=(10,7)

    def setDataFrameFromFile(self,fileName):
        self.df = pd.read_csv(fileName,index_col=None)

    def getDataFrame(self):
        return self.df
    
    def preprocessing(self, target):
        ID_col = 'customerID'
        assert len(CATEGORY_COLS) + len(NUMERIC_COLS) + 2 == self.df.shape[1]

        self.df['TotalCharges']= self.df['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
        self.df['MonthlyCharges'] = self.df['MonthlyCharges'].astype(float)

        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(0)

        self.df['Churn'].replace(to_replace='Yes', value=1, inplace=True)
        self.df['Churn'].replace(to_replace='No',  value=0, inplace=True)

        self.features = self.df.drop(columns=[ID_col, target]).copy()
        self.labels = self.df['Churn'].copy()
        self.features['Churn']=self.labels

    # spliting(features)
    def data_split(self, parameter_split_ratio):

        # data spliting
        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(self.df.iloc[:,:-1],self.df.iloc[:,-1], test_size=(1-parameter_split_ratio), random_state=22)

        #
        ord_enc = OrdinalEncoder()
        ord_enc.fit(self.X_train[CATEGORY_COLS])

        self.X_train_OE = pd.DataFrame(ord_enc.transform(self.X_train[CATEGORY_COLS]), columns=CATEGORY_COLS)
        self.X_train_OE.index = self.X_train.index
        self.X_train_OE = pd.concat([self.X_train_OE, self.X_train[NUMERIC_COLS]], axis=1)

        self.X_test_OE = pd.DataFrame(ord_enc.transform(self.X_test[CATEGORY_COLS]), columns=CATEGORY_COLS)
        self.X_test_OE.index = self.X_test.index
        self.X_test_OE = pd.concat([self.X_test_OE, self.X_test[NUMERIC_COLS]], axis=1)
        print(self.X_train_OE, self.X_test_OE)
    # define classifier
    def predictChurnRate(self, parameter_MaxDepth, parameter_LearningRate, parameter_n_estimators, parameter_colsample_bytree):
        print(parameter_MaxDepth, parameter_LearningRate, parameter_n_estimators, parameter_colsample_bytree)
        # define classifier
        xgbModel = XGBClassifier(max_depth=parameter_MaxDepth,
                                learning_rate=parameter_LearningRate,
                                n_estimators=parameter_n_estimators,
                                verbosity=1,
                                objective='binary:logistic',
                                booster='gbtree',
                                n_jobs=4,
                                gamma=0.001,
                                subsample=0.632,
                                colsample_bytree=parameter_colsample_bytree,
                                colsample_bylevel=1,
                                colsample_bynode=1,
                                reg_alpha=1,
                                reg_lambda=0,
                                scale_pos_weight=40,
                                base_score=0.5,
                                random_state=251162728,
                                missing=1
                                )

        xgbModel.fit(self.X_train_OE, self.y_train)
        self.xgbModel = xgbModel

    def predict(self):
        print(self.xgbModel)
        print(self.X_test_OE)
        self.xgbClassTest = self.xgbModel.predict(self.X_test_OE)

    def predict_proba(self):
        self.xg_probs_test0 = self.xgbModel.predict_proba(self.X_test_OE)
        self.xg_probs_test = self.xg_probs_test0[:, 1]

    def ImportancesFeatures(self):
        # plot features importance
        importances = self.xgbModel.feature_importances_
        indices = np.argsort(importances)[::-1]
        importancePlot, ax = plt.subplots(figsize=self.figsize)
        plt.title("Variable Importance - XGBoosting")
        sns.set_color_codes("pastel")
        sns.barplot(y=[self.X_train_OE.columns[i] for i in indices], x=importances[indices], 
                    label="Total", color="b")
        ax.set(ylabel="Variable",
        xlabel="Variable Importance (Entropy)")
        sns.despine(left=True, bottom=True)
        plt.savefig(self.ImportanceFeaturesImage, format='png')
        self.importancePlot = importancePlot

    def ConfusionMatrix(self):
        confusion_matrix_xgb = confusion_matrix(y_true = self.y_test, y_pred = self.xgbClassTest)
        confusion_matrix_xgb = confusion_matrix_xgb.astype('float') / confusion_matrix_xgb.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(confusion_matrix_xgb, index=['Positive', 'Negative'], columns=['True', 'False'],)
        figsize = self.figsize
        fontsize=14
        # confusion matrix plot
        ConfusionPlot = plt.figure(figsize=figsize)
        heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f')
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,ha='right', fontsize=fontsize)
        plt.savefig(self.confusionMatrixImage, format='png')
        self.confusionPlot = ConfusionPlot
        
        #plot ROC Curve
        # Calculate the ROC curve points
    def ROC_CURVE(self):
        fig = plt.figure(figsize=self.figsize)
        fpr, tpr, thresholds = roc_curve(self.y_test, self.xg_probs_test)
        auc = np.round(roc_auc_score(y_true = self.y_test, y_score = self.xg_probs_test),decimals = 3)
        plt.plot(fpr, tpr,label="AUC - XGBoosting = " + str(auc))
        plt.legend(loc=4)
        plt.savefig(self.roc_curvePlotImage, format='png')
        self.roc_curvePlot = fig

    # multiple ROC plot
    def multiROC(self): # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        skplt.metrics.plot_roc(self.y_test, self.xg_probs_test0,figsize=self.figsize)
        plt.savefig(self.Multiple_ROCPlotImage, format='png')
        
        
    # Report
    def metric_report(self):
        report=classification_report(self.y_test,self.xgbClassTest,output_dict= True)
        self.report = pd.DataFrame(report).transpose() 

    def predict_and_plot(self, parameter_MaxDepth, parameter_LearningRate, parameter_n_estimators, parameter_colsample_bytree):
        self.predictChurnRate(parameter_MaxDepth, parameter_LearningRate, parameter_n_estimators, parameter_colsample_bytree)
        self.predict()
        self.predict_proba()
        self.ImportancesFeatures()
        self.ConfusionMatrix()
        self.ROC_CURVE()
        self.multiROC()
        self.metric_report()
        #self.saved_model()
        

    # Save model
    def saved_model(self):#https://discuss.streamlit.io/t/download-pickle-file-of-trained-model-using-st-download-button/27395
        self.saved_model=self.xgbModel.save_model('XGBoosting1.json')

    #Predicted result
    def predictedResult(self):
        self.X_test_OE['Actual_ChurnRate']=self.y_test
        self.X_test_OE['Predicted_ChurnRate']=self.xgbClassTest
        return self.X_test_OE
        