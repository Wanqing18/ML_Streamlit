import streamlit as st
import pandas as pd
#python -m streamlit run "c:/Users/Gaming/Desktop/Test Code/MLweb/Webapp.py"
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from utils import *
from ChurnRateModel import ChurnRateModel
from scikitplot import metrics
import json
import pickle 

# Page title layout
st.set_page_config(page_title='Churn Rate Prediction With Customized HypterParameters',layout='wide' )

# short introduction
st.header('Churn Rate Prediction With Customized HypterParameters')
st.write(""" This Web aims to predict the churn rate (https://www.investopedia.com/terms/c/churnrate.asp) of a customer using **XGBoosting** with cutomized hyperparameters. 
The end-to-end training process including data preprocessing, Hyperparameter tuning, model fitting, and performance evaluation is completed.""")

# seperate load, profiling, model, ouput into different secction
with st.sidebar:
    st.title('XGBoosting Forecasting')
    st.subheader('Navigation')
    choice=st.radio('Pick your current status',['Upload','Profiling','Modeling','Download'])
    st.info('This application allows you to build an automated ML pipeline Streamlit, Pandas Profiling and Sk-learn')

#if os.path.exists('ChurnRate_Data.csv'):
    #df=pd.read_csv('ChurnRate_Data.csv', index_col= None)

if 'churnRateModel' not in st.session_state:
	st.session_state.churnRateModel = ChurnRateModel()

# upload section
if choice == 'Upload':
    st.title('Upload Data for modeling')
    file = st.file_uploader('Upload Your Dataset Here')
    churnRateModel = st.session_state.churnRateModel
    if file:
        churnRateModel.setDataFrameFromFile(file)
        st.dataframe(churnRateModel.getDataFrame())
        notice=st.info('File has been uploaded')

# profiling section
if choice == 'Profiling':
    st.title('Exploratory Data Analysis')
    st.info('This auto profiling function will provide detailed information for each variable.We could know the type, correlations and distribution of each varaible')
    df = st.session_state.churnRateModel.getDataFrame()
    if st.button('Start Profiling') and not df.empty:
        profile_report=df.profile_report()
        st_profile_report(profile_report)
    else:
        st.info('Please Upload your file first')



# Modeling section
if choice == 'Modeling':# and os.path.exists('ChurnRate_Data.csv'):
    st.title('XGBoosting For Churn Rate Forecasting')
    #if os.path.exists('ChurnRate_Date.csv'):    
    churnRateModel = st.session_state.churnRateModel
    target= st.selectbox('Select Predicted Target', churnRateModel.getDataFrame().columns)
    # Confirm data button (preprocessing)
    if st.button('Start Processing'):
        churnRateModel.preprocessing(target)
        st.info('Data Finished Processing')
    
    if not churnRateModel.features.empty:
        col1,col2 = st.columns(2)
        col1_expander = col1.expander('View Features')
        with col1_expander:
        #click=st.checkbox()
        #if click: # can't show dataframe
            st.dataframe(churnRateModel.features)

        col2_expander = col2.expander('View Labels')
        with col2_expander:
        #click2=st.checkbox('View Labels')
        #if click2:
            st.dataframe(churnRateModel.labels)
    

        # split parameter to left and   result to right
        


        st.markdown(""" <style> .font {font-size:11px ; font-family: 'serif'; color: #17202A;} </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Set Parameters</p>',unsafe_allow_html=True)
        # set split ratio
        st.subheader('Split Ratio')#,max_depth,n_estimator,colsample_bytree
        parameter_split_ratio=st.slider('Insert split ratio(% for Training set)', step=0.1,min_value=0.6, max_value=1.0)
        st.write('The current split ratio is ', parameter_split_ratio)
        # confirm spliting ratio
        if st.button('Start spliting'):
            churnRateModel.data_split(parameter_split_ratio)
            st.info('Data Finished Spliting')
        if not churnRateModel.X_train_OE.empty: 
            # adjust layout
            
            # set learning rate
        
            st.subheader('Learning Rate')#,max_depth,n_estimator,colsample_bytree
            parameter_LearningRate=st.number_input('Insert learning rate',step=0.001,min_value=0.000, max_value=1.000 )
            st.write('The current learning rate is ', parameter_LearningRate)
            
            # set max_depth
        
            st.subheader('Max Depth')#,max_depth,n_estimator,colsample_bytree
            parameter_MaxDepth=st.number_input('Insert Max Depth',min_value=0)
            st.write('The current Max Depth of a tree is ', parameter_MaxDepth)
            #st.info(' Increasing this value will make the model more complex and more likely to overfit.')

        
            #set n_estimators
        
            st.subheader('Numbers of Estimators')#,max_depth,n_estimator,colsample_bytree
            parameter_n_estimators=st.slider('Insert n_estimators',min_value=10,max_value=150,step=5)
            st.write('The current Numbers of Estimators are ', parameter_n_estimators)

            # set colsample_bytree
        
            st.subheader('Colsample_bytree')#,max_depth,n_estimator,colsample_bytree
            parameter_colsample_bytree=st.slider('Insert colsample_bytree',min_value=0.0,max_value=1.0,step=0.1)
            st.write('The current colsample_bytree are ', parameter_colsample_bytree)
            #st.info('this is the subsample ratio of columns when constructing each tree.')

            # start predict
        
            if st.button('Predict') and not churnRateModel.getDataFrame().empty:
                churnRateModel.predict_and_plot(parameter_MaxDepth, parameter_LearningRate, parameter_n_estimators, parameter_colsample_bytree)
                st.info('Finished Prediction')


            # display result    
            result=st.multiselect('Select the result you want to see',['ROC_Curve','Multiple_ROC','Metric_Report','ConfusionMatrix','ImportanceFeatures','PredictedOutcome'])
            col1,col2 = st.columns(2)
            with col1:
                if 'ROC_Curve' in result:
                    st.subheader('ROC_Curve')
                    st.pyplot(churnRateModel.roc_curvePlot)
                    
                if 'Multiple_ROC' in result: # special cases: Axessubplot
                    st.subheader('Multiple_ROC')
                    multiplot = metrics.plot_roc(churnRateModel.y_test, churnRateModel.xg_probs_test0).figure
                    multiplot

                if 'Metric_Report' in result:
                    st.subheader('Metric_Report')
                    report=churnRateModel.report
                    st.dataframe(report)

            with col2:
                if 'ConfusionMatrix' in result:
                    st.subheader('ConfusionMatrix')
                    st.pyplot(churnRateModel.confusionPlot)
                    
                if 'ImportanceFeatures' in result:
                    st.subheader('ImportanceFeatures')
                    st.pyplot(churnRateModel.importancePlot)

            if 'PredictedOutcome' in result:
                st.subheader('PredictedOutcome')
                result=churnRateModel.predictedResult()
                st.dataframe(result)
            


# download section:display result and download
if choice == 'Download':
    st.title('Download Ouput Here')
    # create multi label select to see result
    # download selected result and saved model
    churnRateModel = st.session_state.churnRateModel
    options=st.multiselect('Select the result you want to download',['ROC_Curve','Multiple_ROC','Metric_Report','ConfusionMatrix','ImportanceFeatures','Model','PredictedOutcome'])
    
    #display layout
    d_col1,d_col2 = st.columns(2)

    with d_col1:
        if 'ROC_Curve' in options:
            st.subheader('ROC_Curve')
            st.pyplot(churnRateModel.roc_curvePlot) 
            fn_roc='ROC_Curve.png'
            btn_roc = st.download_button(label="Download ROC_Curve Graph",data=churnRateModel.roc_curvePlotImage,file_name=fn_roc,mime="image/png")
        
        if 'Multiple_ROC' in options: # special cases: Axessubplot
            st.subheader('Multiple_ROC')
            multiplot = metrics.plot_roc(churnRateModel.y_test, churnRateModel.xg_probs_test0).figure
            multiplot
            fn_m=('Multiple_ROC')
            btn_rocm = st.download_button(label="Download Multiple_ROC Graph",data=churnRateModel.Multiple_ROCPlotImage,file_name=fn_m,mime="image/png")

        if 'Metric_Report' in options:
            st.subheader('Metric_Report')
            report=churnRateModel.report
            st.dataframe(report)
            st.download_button(label='Download Metric_Report', data = report.to_csv(index=False), file_name='Metric_Report.csv', mime='text/csv')

    with d_col2:
        if 'ConfusionMatrix' in options:
            st.subheader('ConfusionMatrix')
            st.pyplot(churnRateModel.confusionPlot)
            fn='ConfusionMatrix.png'
            btn = st.download_button(label="Download Confusion Matrix Graph", data=churnRateModel.confusionMatrixImage, file_name=fn, mime="image/png")

        if 'ImportanceFeatures' in options:
            st.subheader('ImportanceFeatures')
            st.pyplot(churnRateModel.importancePlot)
            fn_if='ImportanceFeatures.png'
            btn_if = st.download_button(label="Download ImportanceFeatures Graph", data=churnRateModel.ImportanceFeaturesImage,file_name=fn_if,mime="image/png")

    if 'Model' in options:
        st.write("XGBoosting Model has been saved as pkl file and ready for download!:smiley:")
        st.download_button(label='Download Model', data = pickle.dumps(churnRateModel.xgbModel), file_name='model.pkl')

    if 'PredictedOutcome' in options:
        result=churnRateModel.predictedResult()
        st.dataframe(result)
        st.download_button(label='Download PredictedOutcome', data = result.to_csv(index=False), file_name='PredictedOutcome.csv', mime='text/csv')

    #st.download_button('Download Selected Result', data=options,filename=,mime=image/png)
   

