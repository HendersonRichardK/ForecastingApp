import pandas as pd
import numpy as np
import os
import pyodbc
import time
import datetime
import pandas_profiling
import yfinance as yf
import fbprophet
import streamlit as st
import plotly
import seaborn as sns
import requests
import chart_studio.plotly as py
import plotly.graph_objects as go
import cufflinks as cf
import pybase64
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import  plot_plotly, plot_components_plotly
from plotly import graph_objects as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from fbprophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import LabelEncoder
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from dotenv import load_dotenv
from datetime import date, timedelta
from pandas_profiling import ProfileReport
from streamlit_lottie import st_lottie
from streamlit_pandas_profiling import st_profile_report

from enum import Enum
from io import BytesIO, StringIO



import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="centered")
##st.set_page_config(page_title="Automated Time Series Forecasting", page_icon="📈")

plt.rcParams["figure.figsize"] = (15,8)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_home = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_XyfCyq.json')
st_lottie(lottie_home, speed=1.5, width = 800, height =400)

with st.sidebar.header('Pandas Profiling Report'):
    pr_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Or just download this example CSV and try](👆 Upload CSV or  Sample: [rate.csv](https://fairwaymtg-my.sharepoint.com/:x:/r/personal/rhenderson_fairwaymc_com/Documents/rate.csv?d=w133a1104030342029f6acbb0355975cb&csf=1&web=1&e=ru2vVp))  """)

# Pandas Profiling Report
if pr_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(pr_file)
        return csv
    pr_df = load_csv()
    profile_rpt = ProfileReport(pr_df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(pr_df)
    st.write('---')
    st.header('**Exploratory Data Analysis Report for the Input Dataset**')
    st_profile_report(profile_rpt)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to create an Example Dataset and explore the auto-generated insights'):
        # Example data
        @st.cache
        def load_data():
            df_pd = pd.DataFrame(
                np.random.rand(100, 3),
                columns=['Variable #1', 'Variable #2', 'Variable #3']
                )
            df_pd['Are Getting Rates Better?'] = np.random.choice(['Yes', 'No'], len(df_pd))
            df_pd['Who is to Blame'] = np.random.choice(['Iron Man', 'Ace Ventura', 'Lahey', 'Elon Musk'], len(df_pd))
            df_pd['Label'] = np.random.randint(0, 2, len(df_pd))
            return df_pd
        dataframe = load_data()
        pr_df = ProfileReport(dataframe, explorative=True)
        st.header('**Input DataFrame**')
        st.write(dataframe)
        st.write('---')
        st.header('**Exploratory Data Analysis Report for the Example Dataset**')
        st_profile_report(pr_df)

st.title('Automated Time Series Forecasting')

# creating file uploading mech:
uploaded_file = st.file_uploader("Chose a CSV file",type = 'CSV')
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    # creating a checkbox to show dataframe:
    if st.checkbox('Show DataFrame'):
        st.write(dataset)

# for disabling the warning:
st.set_option('deprecation.showfileUploaderEncoding', False)

#for modifying model
modify_options = ('Changepoints range','changepoint prior scale','Growth','changepoint numbers','Yearly seasonality','Weekly seasonality','daily seasonality','seasonality mode','seasonality prior scale','MCMC samples','Interval width','Uncertainty samples')
modify = st.sidebar.multiselect('select option to edit',modify_options)

period = st.number_input('Select Period',value = 0)
st.write("Period : ",period)
freq_option_dic = {'options_freq':['Seconds','minute','hourly','daily','weekly','monthly','quaterly','yearly']  ,'options_code':['S','M','H','D','W','MS','Q','Y'] }
freq_option_df = pd.DataFrame(freq_option_dic)
freq_selected = st.selectbox('Choose Frequency',freq_option_df['options_freq'])
index = 0
for opt in freq_option_df['options_freq']:
    if freq_selected == opt:
        freq_code = freq_option_df.iloc[index,1]
    index = index + 1
st.write("Frequency : ",freq_selected)

#for changing Changepoints
change_point_value = 0.8
if 'Changepoints range' in modify:
    change_point_value = st.sidebar.slider('Select value for change point:',0.0,1.0,0.8,0.05)


#for changing changepoint_prior_scale
prior_scale=0.05
if 'changepoint prior scale' in modify:
    prior_scale = st.sidebar.slider('Select value for changepoint prior scale:',0.0,1.0,0.05,0.01)

#for chaning growth
growth_selected = "linear"
if 'Growth'in modify:
    growth_options = ["liner","logistic"]
    growth_selected = st.sidebar.radio('select growth',growth_options)

#for changing changepoint numbers
changepoint_number = 25
if 'changepoint numbers' in modify:
    changepoint_number = st.sidebar.number_input('select number of changepoints',1,100,1)

#for changing seasonality
seasonality_options = ["auto",True,False,"Select yourself"]
yearly_seasonality_selected = "auto"
if 'Yearly seasonality' in modify:
    yearly_option_selected = st.sidebar.radio("Choose and opions",seasonality_options)
    if yearly_option_selected == "Select yourself":
        yearly_seasonality_selected = st.sidebar.number_input("Enter number of Fourier terms")
    else:
        yearly_seasonality_selected = yearly_option_selected

weekly_seasonality_selected = "auto"
if 'Weekly seasionality' in modify:
    yearly_option_selected = st.sidebar.radio("Choose and opions",seasonality_options)
    if weekly_option_selected == "Select yourself":
        weekly_seasonality_selected = st.sidebar.number_input("Enter number of Fourier terms")
    else:
        weekly_seasonality_selected = weekly_option_selected

daily_seasonality_selected = "auto"
if 'daily seasonality' in modify:
    daily_option_selected = st.sidebar.radio("Choose and opions",seasonality_options)
    if daily_option_selected == "Select yourself":
        daily_seasonality_selected = st.sidebar.number_input("Enter number of Fourier terms")
    else:
        daily_seasonality_selected = daily_option_selected

#for changing seasonality mode
seasonality_mode_selected = "additive"
if 'seasonality mode'in modify:
    seasonality_mode_options = ["additive","multiplicative"]
    seasonality_mode_selected = st.sidebar.radio("select seasonality mode",seasonality_mode_options)

#for changing seasonality prior scale
seasonality_prior_scale_selected = 10
if 'seaonsality prior scale' in modify:
    seasonality_prior_scale_selected = st.sidebar.number_input("select seasonality prior scale",value=10)

#for changing mcmc samples
mcmc_samples_selected = 0
if 'MCMC samples' in modify:
    mcmc_samples_selected = st.sidebar.number_input("select a number",value=0)

#for chaning inteval width
interval_width_selected = 0.8
if 'Interval width' in modify:
    interval_width_selected = st.sidebar.slider("select interval width",0.0,1.0,0.8,0.1)

#for changing uncertainty samples
uncertainty_samples_selected = 1000
if 'Uncertainty samples' in modify:
    uncertainty_samples_selected = st.sidebar.number_input("select number of uncertainty samples",value=1000)

st.write('\n\n')
st.write("Hyperparameters tuning can be done in side menu")
st.write("*Please note that changing hyperparameters may cause unwanted changes so proceed with precautions")

# buidling model
if st.button('Run'):
    model = Prophet(
        growth = growth_selected,
        n_changepoints = changepoint_number,
        changepoint_range = change_point_value,
        changepoint_prior_scale = prior_scale,
        yearly_seasonality = yearly_seasonality_selected,
        weekly_seasonality = weekly_seasonality_selected,
        seasonality_mode = seasonality_mode_selected,
        seasonality_prior_scale = seasonality_prior_scale_selected,
        mcmc_samples = mcmc_samples_selected,
        interval_width = interval_width_selected,
        uncertainty_samples = uncertainty_samples_selected
        ).fit(dataset)
    future = model.make_future_dataframe(periods = period, freq = freq_code)
    future_dataframe = model.predict(future)
    fig = plot_plotly(model, future_dataframe)
    st.write(fig)
    fig2 = model.plot_components(future_dataframe)
    st.write(fig2)
    fig3 = model.plot(future_dataframe)
    a = add_changepoints_to_plot(fig3.gca(), model, future_dataframe)
    st.write(fig3)
    # fig3 = plot_components_plotly(model, future_dataframe)
    # st.write(fig3)


