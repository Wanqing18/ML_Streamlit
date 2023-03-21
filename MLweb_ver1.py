import streamlit as st
import pandas as pd
#python -m streamlit run "./Webapp1.py"
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
# hide main menue and customized footer for internal user
hide_menu="""
<style>
#MainMenu {
    visibility:hidden;}
footer{
        visibility:visible;
}
footer:after{
        content:'Copyright @ 2023 : Wanqing';
        display: block;
        position: relative;
}
</style>
"""

# Page title layout
st.set_page_config(page_title='Churn Rate Prediction With Customized HypterParameters',layout='centered' )

# Load customized text style for each title
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# short introduction
#st.subheader('Churn Rate Prediction With Customized HypterParameters')
#st.markdown("<h4>This Web aims to predict the churn rate of a customer using **XGBoosting** with cutomized hyperparameters.</h4>", unsafe_allow_html=True) 
#The end-to-end training process including data preprocessing, Hyperparameter tuning, model fitting, and performance evaluation is completed.""")

st.markdown(hide_menu,unsafe_allow_html = True)
# seperate load, profiling, model, ouput into different secction
with st.sidebar:
    st.title('Churn Rate Forecasting')
    st.subheader('Navigation')
    choice=st.radio('Choose Your Status',['Upload','Profiling','Modeling','Download'])
    st.info('Through following the steps above, you would be able to complete the EDA, XGBmodel training, and download the predicted result and evaluation metrics.')

if 'churnRateModel' not in st.session_state:
	st.session_state.churnRateModel = ChurnRateModel()
            
if 'Predicted' not in st.session_state: # set a flag for output display and download
    st.session_state.Predicted = False 

if "parameter_LearningRate" not in st.session_state:
    st.session_state.parameter_LearningRate = False

if "parameter_MaxDepth" not in st.session_state:
    st.session_state.parameter_MaxDepth = False

if "parameter_colsample_bytree" not in st.session_state:
    st.session_state.parameter_colsample_bytree = False
    
# upload section
if choice == 'Upload':
    st.markdown(f'<h1> Upload Data For Modeling</h1>',unsafe_allow_html=True)
    st.markdown(f'<h3> Upload Your Dataset Here</h3>',unsafe_allow_html=True)
    file = st.file_uploader('Upload Your Dataset Here',type=["csv"],label_visibility="collapsed")
    churnRateModel = st.session_state.churnRateModel
    if file:
        churnRateModel.setDataFrameFromFile(file)
        st.dataframe(churnRateModel.getDataFrame())
        st.info('File has been uploaded. You could go to the **Profiling** session or the **Modeling** session now.')

# profiling section
if choice == 'Profiling':
    #st.header('Exploratory Data Analysis')
    st.markdown(f'<h1> Exploratory Data Analysis</h1>',unsafe_allow_html=True)
    st.write("Interactive reports of the dataset would be generated here. The correlations of each variable and information for missing values could also be visualized and downloaded. ")
    df = st.session_state.churnRateModel.getDataFrame()
    if st.button('Start Profiling'):
        if not df.empty:     
            profile_report=df.profile_report()
            st_profile_report(profile_report)
            export=profile_report.to_html()
            st.download_button(label="Download Full Profiling Report", data=export, file_name='Profiling_Report.html')     
        else:
            st.info('Please Upload Your File First.')

# Modeling section
if choice == 'Modeling':
    #st.title('XGBoosting For Churn Rate Forecasting')
    st.markdown('<h1>XGBoosting For Churn Rate Forecasting</h1>',unsafe_allow_html=True)
    st.write('Data preprocessing, Hyperparameter tuning, model fitting, and performance evaluation would be completed in this page.')
    #if os.path.exists('ChurnRate_Date.csv'):    
    churnRateModel = st.session_state.churnRateModel
    st.markdown('<h3>Select Predicted Target:</h3>',unsafe_allow_html=True)
    target= st.selectbox('Select Predicted Target',churnRateModel.getDataFrame().columns, label_visibility="collapsed")
    if target:
        st.write(target, 'has been selected for prediction.')

    # Confirm data button (preprocessing)
    if st.button('Start Processing'): 
        if not churnRateModel.getDataFrame().empty:     
            churnRateModel.preprocessing(target)
            
        else:
            st.info('Please Upload Your File First.')
         
    if not churnRateModel.features.empty: 
        st.info('Data Finished Processing.')
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
    
        
        # set split ratio
        st.markdown('<h1 style="font-size: 26px;line_hight:2">Train-Test Split</h1>',unsafe_allow_html=True)
        parameter_split_ratio=st.slider('Insert split ratio for Training set', step=0.1,min_value=0.5, max_value=0.9,label_visibility="collapsed")
        st.write('The current split ratio is ', parameter_split_ratio, 'for training dataset.')
        # confirm spliting ratio
        if st.button('Start Splitting'):
            churnRateModel.data_split(parameter_split_ratio)
            st.info('Data Finished Spliting.')
        
        
        if not churnRateModel.X_train_OE.empty: 
            # adjust layout
            #st.markdown(""" <style> .font {font-size:30px ;  color: #17202A;} </style> """, unsafe_allow_html=True)
            st.markdown('''<h1 style="font-size: 26px">Set Parameters</h1>''',unsafe_allow_html=True)
            st.write("Spliting Ratio, Learning Ratio, Max Depth, Numbers of Estimators and Colsample_bytree could be selected.")
            # set learning rate
            st.markdown('<h2>Learning Rate</h2>',unsafe_allow_html=True)
            st.session_state.parameter_LearningRate=st.number_input('Insert learning rate',step=0.001,min_value=0.000, max_value=1.000,label_visibility="collapsed" )
            st.write('The current learning rate is ', st.session_state.parameter_LearningRate)
            if  (st.session_state.parameter_LearningRate == 0):
                    st.info('Please Input parameter_LearningRate.')

            # set max_depth
            st.markdown('<h2>Max Depth</h2>',unsafe_allow_html=True)
            st.session_state.parameter_MaxDepth=st.number_input('Insert Max Depth',min_value=0,label_visibility="collapsed")
            st.write('The current Max Depth of a tree is ', st.session_state.parameter_MaxDepth)
            #st.info(' Increasing this value will make the model more complex and more likely to overfit.')
            if  (st.session_state.parameter_MaxDepth == 0):
                    st.info('Please Select Max_Depth.')
        
            #set n_estimators
            st.markdown('<h2>Numbers of Estimators</h2>',unsafe_allow_html=True)
            parameter_n_estimators=st.slider('Insert n_estimators',min_value=10,max_value=150,step=5,label_visibility="collapsed")
            st.write('The current Numbers of Estimators are ', parameter_n_estimators)

            # set colsample_bytree
            st.markdown('<h2>Colsample_bytree</h2>',unsafe_allow_html=True)
            st.session_state.parameter_colsample_bytree=st.slider('Insert colsample_bytree',min_value=0.0,max_value=1.0,step=0.1,label_visibility="collapsed")
            st.write('The current colsample_bytree are ', st.session_state.parameter_colsample_bytree)
            #st.info('this is the subsample ratio of columns when constructing each tree.')
            if (st.session_state.parameter_colsample_bytree == 0):
                    st.info('Please Select parameter_colsample_bytree.')
            # start predict with all parameters get selected
         
            #if (parameter_LearningRate == 0) and (parameter_MaxDepth == 0) and (parameter_colsample_bytree == 0):
                #st.info('Please Select Your parameters.')       

            if (st.session_state.parameter_LearningRate != 0) and (st.session_state.parameter_MaxDepth != 0) and (st.session_state.parameter_colsample_bytree != 0):   
                if st.button('Predict'):
                        churnRateModel.predict_and_plot(st.session_state.parameter_MaxDepth, st.session_state.parameter_LearningRate, parameter_n_estimators, st.session_state.parameter_colsample_bytree)
                        st.info('Finished Prediction.')
                        st.session_state.Predicted= True # change it to global

                if st.session_state.Predicted: 
                    st.markdown('<h3>Select the result you want to see:</h3>',unsafe_allow_html=True)
                    result=st.multiselect('Select the result you want to see',['ROC_Curve','Multiple_ROC','Metric_Report','ConfusionMatrix','ImportanceFeatures','PredictedOutcome'],label_visibility="collapsed")

                    col1,col2 = st.columns(2)
                    with col1:
                        if 'ROC_Curve' in result:
                            st.subheader('ROC_Curve')
                            st.pyplot(churnRateModel.roc_curvePlot)
                            
                        if 'Metric_Report' in result:
                            st.subheader('Metric_Report')
                            report=churnRateModel.report
                            st.dataframe(report)

                        if 'ImportanceFeatures' in result:
                            st.subheader('ImportanceFeatures')
                            st.pyplot(churnRateModel.importancePlot)

                    with col2:
                        if 'Multiple_ROC' in result: # special cases: Axessubplot
                            st.subheader('Multiple_ROC')
                            multiplot = metrics.plot_roc(churnRateModel.y_test, churnRateModel.xg_probs_test0).figure
                            multiplot

                        if 'ConfusionMatrix' in result:
                            st.subheader('ConfusionMatrix')
                            st.pyplot(churnRateModel.confusionPlot)

                    if 'PredictedOutcome' in result:
                        st.subheader('PredictedOutcome')
                        result=churnRateModel.predictedResult()
                        st.dataframe(result)
                


# download section:display result and download
if choice == 'Download':
    #st.title('Download Ouput')
    st.markdown(f'<h1> Download Ouput</h1>',unsafe_allow_html=True)
    # create multi label select to see result
    # download selected result and saved model
    churnRateModel = st.session_state.churnRateModel
    if st.session_state.Predicted:
        st.markdown(f'<h3> Select the result you want to download.</h3>',unsafe_allow_html=True)
        options=st.multiselect('Select the result you want to download',['ROC_Curve','Multiple_ROC','Metric_Report','ConfusionMatrix','ImportanceFeatures','Model','PredictedOutcome',],label_visibility="collapsed")

        #display layout
        d_col1,d_col2 = st.columns(2)

        with d_col1:
            if 'ROC_Curve' in options:
                st.subheader('ROC_Curve')
                st.pyplot(churnRateModel.roc_curvePlot) 
                fn_roc='ROC_Curve.png'
                btn_roc = st.download_button(label="Download ROC_Curve Graph",data=churnRateModel.roc_curvePlotImage,file_name=fn_roc,mime="image/png")
            
            if 'Metric_Report' in options:
                st.subheader('Metric_Report')
                report=churnRateModel.report
                st.dataframe(report)
                st.download_button(label='Download Metric_Report', data = report.to_csv(index=False), file_name='Metric_Report.csv', mime='text/csv')
            
            if 'ImportanceFeatures' in options:
                st.subheader('ImportanceFeatures')
                st.pyplot(churnRateModel.importancePlot)
                fn_if='ImportanceFeatures.png'
                btn_if = st.download_button(label="Download ImportanceFeatures Graph", data=churnRateModel.ImportanceFeaturesImage,file_name=fn_if,mime="image/png")

        with d_col2:
            if 'Multiple_ROC' in options: # special cases: Axessubplot
                st.subheader('Multiple_ROC')
                multiplot = metrics.plot_roc(churnRateModel.y_test, churnRateModel.xg_probs_test0).figure
                multiplot
                fn_m=('Multiple_ROC')
                btn_rocm = st.download_button(label="Download Multiple_ROC Graph",data=churnRateModel.Multiple_ROCPlotImage,file_name=fn_m,mime="image/png")

            if 'ConfusionMatrix' in options:
                st.subheader('ConfusionMatrix')
                st.pyplot(churnRateModel.confusionPlot)
                fn='ConfusionMatrix.png'
                btn = st.download_button(label="Download Confusion Matrix Graph", data=churnRateModel.confusionMatrixImage, file_name=fn, mime="image/png")

            
        if 'Model' in options:
            st.write("XGBoosting Model has been saved as pkl file and ready for download!:smiley:")
            st.download_button(label='Download Model', data = pickle.dumps(churnRateModel.xgbModel), file_name='model.pkl')

        if 'PredictedOutcome' in options:
            result=churnRateModel.predictedResult()
            st.dataframe(result)
            st.download_button(label='Download PredictedOutcome', data = result.to_csv(index=False), file_name='PredictedOutcome.csv', mime='text/csv')

    else:
        st.info('Please Finish Your Model Training First.')
   

