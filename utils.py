# def preprocessing(dataframe):

#     assert len(category_cols) + len(numeric_cols) + 2 == dataframe.shape[1]


#     dataframe['TotalCharges']= dataframe['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
#     dataframe['MonthlyCharges'] = dataframe['MonthlyCharges'].astype(float)

#     dataframe['TotalCharges'] = dataframe['TotalCharges'].fillna(0)

#     dataframe['Churn'].replace(to_replace='Yes', value=1, inplace=True)
#     dataframe['Churn'].replace(to_replace='No',  value=0, inplace=True)

#     features = dataframe.drop(columns=[ID_col, target]).copy()
#     labels = dataframe['Churn'].copy()
#     # view result(optional)
#     st.dataframe(features)
#     st.dataframe(labels)
#     features['Churn']=labels
#     return features

# def data_split(df):
    
#         # data spliting
#         X_train, X_test,y_train,y_test= train_test_split(df.iloc[:,:-1],df.iloc[:,-1], test_size=(1-Parameter_splitratio), random_state=22)

#         #
#         ord_enc = OrdinalEncoder()
#         ord_enc.fit(X_train[category_cols])

#         X_train_OE = pd.DataFrame(ord_enc.transform(X_train[category_cols]), columns=category_cols)
#         X_train_OE.index = X_train.index
#         X_train_OE = pd.concat([X_train_OE, X_train[numeric_cols]], axis=1)

#         X_test_OE = pd.DataFrame(ord_enc.transform(X_test[category_cols]), columns=category_cols)
#         X_test_OE.index = X_test.index
#         X_test_OE = pd.concat([X_test_OE, X_test[numeric_cols]], axis=1)

#         st.dataframe(X_test_OE)
#         st.dataframe(X_train_OE)
#         return X_test_OE, X_train_OE,y_train,y_test

# # define classifier
# def PredictChurnRate(data):

#     # define classifier
#     odf_xgb = XGBClassifier(max_depth=Parameter_MaxDepth,
#                             learning_rate=Parameter_LearningRate,
#                             n_estimators=Parameter_n_estimators,
#                             verbosity=1,
#                             objective='binary:logistic',
#                             booster='gbtree',
#                             n_jobs=4,
#                             gamma=0.001,
#                             subsample=0.632,
#                             colsample_bytree=Parameter_colsample_bytree,
#                             colsample_bylevel=1,
#                             colsample_bynode=1,
#                             reg_alpha=1,
#                             reg_lambda=0,
#                             scale_pos_weight=40,
#                             base_score=0.5,
#                             random_state=251162728,
#                             missing=None
#                             )
                    

#     odf_xgb.fit(X_train_OE,y_train)
#     # show result
#     # plot features importance
#     importances = odf_xgb.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     f, ax = plt.subplots(figsize=(3, 8))
#     plt.title("Variable Importance - XGBoosting")
#     sns.set_color_codes("pastel")
#     sns.barplot(y=[X_train_OE.columns[i] for i in indices], x=importances[indices], 
#                 label="Total", color="b")
#     ax.set(ylabel="Variable",
#     xlabel="Variable Importance (Entropy)")
#     sns.despine(left=True, bottom=True)

#     # plot confusion matrix
#     XGBClassTest = odf_xgb.predict(X_test_OE)
#     xg_probs_test = odf_xgb.predict_proba(X_test_OE)
#     xg_probs_test = xg_probs_test[:, 1]


#     confusion_matrix_xgb = confusion_matrix(y_true = y_test, 
#                         y_pred = XGBClassTest)


#     confusion_matrix_xgb = confusion_matrix_xgb.astype('float') / confusion_matrix_xgb.sum(axis=1)[:, np.newaxis]


#     df_cm = pd.DataFrame(
#             confusion_matrix_xgb, index=['good', 'bad'], columns=['good', 'bad'], 
#     )


#     figsize = (10,7)
#     fontsize=14


#     fig = plt.figure(figsize=figsize)
#     heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f')


#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, 
#                                 ha='right', fontsize=fontsize)
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
#                                 ha='right', fontsize=fontsize)


# # show metrics report
# report=classification_report(y_test,XGBClassTest,digits=4)

CATEGORY_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                    'PaymentMethod']

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
