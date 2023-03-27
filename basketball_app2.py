import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import requests
from streamlit_option_menu import option_menu
from google.oauth2 import service_account
from gsheetsdb import connect
from gspread_pandas import Spread,Client
    
st.title('NBA Player Stats')

#Top menu
selected_page = option_menu(
    menu_title=None,
    options=["Player Stats","Team Stats"],
    default_index=0,
    orientation='horizontal'
)

st.markdown("""
This is my model. Here's how to use it.
""")

#SECTION: GETTING, CLEANING AND FILTERING DATA

@st.experimental_memo #recent
def load_data2(): #recent
    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    conn = connect(credentials=credentials)
    sheet_url = st.secrets["private_gsheets_url"]
    query = f'SELECT * FROM "{sheet_url}"'
    rows = conn.execute(query, headers=0).fetchall()

    # Convert data to a Pandas DataFrame.
    df2 = pd.DataFrame.from_records(rows, columns=rows[0])
    return df2 #recent
df2 = load_data2() #recent

#Rename columns
df2.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
df2.rename(columns={'0':'Scenario','1':'Year','2':'Player','3':'Position','4':'Age','5':'Team','6':'G','7':'GS','8':'MP','9':'MP/G','10':'Scoring','11':'Passing','12':'Rebounds','13':'Total Offense','14':'Total Defense','15':'Total Score','16':'MP Threshold'},inplace=True)
#Make data type appropriate
df2['Year'] = df2['Year'].astype(int)
df2['Age'] = df2['Age'].astype(int)
df2['G'] = df2['G'].astype(int)
df2['GS'] = df2['GS'].astype(int)
df2['MP'] = df2['MP'].astype(int)

df2.drop('MP Threshold',axis=1,inplace=True) #Delete unneeded column.
    
#User input year and filtering model data by selected year (Sidebar). This needs to be above load_data, a function of year.
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2024))))
df_exp = df2
df2 = df2[df2['Year']==selected_year] #Filter by selected year

# Web scraping of NBA player stats from basketball-reference.com
@st.cache()
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    df1 = raw.drop(['Rk'], axis=1)
    return df1
df1 = load_data(selected_year)

def load_data_exp(year):

    url = 'https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_standings.html'

    # Read the HTML table into a list of DataFrames
    dfs = pd.read_html(url)

    # Select the second DataFrame (index 1)
    df0 = dfs[0]
#    df1 = df[1]
#    df = pd.concat([df0,df1])

    # Select only the columns we want
    df = df0['W']

    return df

@st.cache()
def load_data_playoffs(year):
    url = "https://www.basketball-reference.com/playoffs/NBA_" + str(year) + "_per_game.html"
    response = requests.get(url)
    if response.status_code == 404:
        return None
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    df1 = raw.drop(['Rk'], axis=1)
    return df1
dfTemp = load_data_playoffs(selected_year)
if load_data_playoffs(selected_year) is not None:
    dfP = load_data_playoffs(selected_year)

# User input Position selection and filtering data by positions
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)
df2 = df2[df2.Position.isin(selected_pos)]
df1 = df1[df1.Pos.isin(selected_pos)]

#Select regular season or playoffs and filter data accordingly
list_of_scenarios = ['Regular Season']
if 'Playoffs' in set(df2['Scenario']): #ie if the year has gotten to the playoffs yet.
    list_of_scenarios.append('Playoffs')
scenario = st.sidebar.selectbox('Scenario',list_of_scenarios)
df2 = df2[df2['Scenario']==scenario] #Filter by scenario

# Sidebar - Team selection
sorted_unique_team = sorted(df2.Team.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
df1 = df1[df1.Tm.isin(selected_team)]
df2 = df2[df2.Team.isin(selected_team)]

#Minutes Played Filter (We don't always want players with low minutes to be included)
max_games_played = min([82,max(df2['G'])]) #Roughly how many games have been played in the season so far? 
default_mins = int((max_games_played/82)*1200)
if scenario=='Playoffs': #In the playoffs we start with 0 as the minimum since lots of teams play very few games.
    mp = int(st.sidebar.text_input('Minimum Minutes Played','0'))
else:
    mp = int(st.sidebar.text_input('Minimum Minutes Played',str(default_mins)))
df2 =df2[df2["MP"] >= mp] #Filter df2 by MP. 

#Sort data by Total Score and reindex.j
df2.sort_values(by=['Total Score'],axis=0,ascending=False,inplace=True)
df2.reset_index(drop=True,inplace=True)
df2.index = np.arange(1, len(df2) + 1)

#SECTION: DATA EXPLORATION

#Team Data function. 
def byTeam():
    data = []
    for i in range(len(selected_team)):
        filt = df2['Team']==selected_team[i]
        df = df2[filt]
        if df['MP'].sum()==0:
            (avgTot,avgOff,avgDef,avgPass,avgScore,avgRebound,avgAge)=(0,0,0,0,0,0,0)
        else:
            avgTot = ((df['Total Score']*df['MP']).sum())/(df['MP'].sum()) #Weighted average of total weighted by MP. 
            avgOff = ((df['Total Offense']*df['MP']).sum())/(df['MP'].sum()) #Weighted avg of offense weighted by MP.
            avgDef = ((df['Total Defense']*df['MP']).sum())/(df['MP'].sum()) #Weighted avg of defense weighted by MP. 
            avgPass = ((df['Passing']*df['MP']).sum())/(df['MP'].sum())
            avgScore = ((df['Scoring']*df['MP']).sum())/(df['MP'].sum())
            avgRebound = ((df['Rebounds']*df['MP']).sum())/(df['MP'].sum())
            avgAge = ((df['Age']*df['MP']).sum())/(df['MP'].sum())
        data.append((selected_team[i],avgTot,avgOff,avgDef,avgPass,avgScore,avgRebound,avgAge))
    df = pd.DataFrame(data)
    df.columns = ['0','1','2','3','4','5','6','7']
    df.rename(columns = {'0':'Team','1':'Avg Total','2':'Avg Off','3':'Avg Def','4':'Avg Passing','5':'Avg Scoring','6':'Avg Rebounds','7':'Avg Age'},inplace=True)
    return df

def byTeam_exp():
    data = []
    for i in range(len(selected_team)):
        filt = df_exp['Team']==selected_team[i]
        df = df_exp[filt]
        if df['MP'].sum()==0:
            avgTot=0
        else:
            avgTot = ((df['Total Score']*df['MP']).sum())/(df['MP'].sum()) #Weighted average of total weighted by MP. 
            # avgOff = ((df['Total Offense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum()) #Weighted avg of offense weighted by MP.
            # avgDef = ((df['Total Defense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum()) #Weighted avg of defense weighted by MP. 
            # avgPass = ((df['Passing']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
            # avgScore = ((df['Scoring']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
            # avgRebound = ((df['Rebounds']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
            # avgAge = ((df['Age']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
        data.append((selected_team[i],avgTot))
    df = pd.DataFrame(data)
    df.columns = ['0','1']
    df.rename(columns = {'0':'Team','1':'Avg Total'},inplace=True)
    return df

players = [] #Players are manually selected (via user search) in line 154
def byPlayer():
    filt = df2.Player.isin(players)
    df = df2[filt]
    return df
#
def byX(year,teams,positions,mp,x): #A team stat that graphs by X like below. NEEDS WORK *&$#(* &^# *&^# (*&#^ )*$&^()*&@ )*(&^# )(*^&@)
    listt = []
    for i in range(min(df[x])):
        filt = df2['Age']==i
        df = df2[filt]
        avgTot = ((df['Total Score']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
        avgOff = ((df['Total Offense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
        avgDef = ((df['Total Defense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
        listt.append((teams[i],avgTot,avgOff,avgDef))
    df = pd.DataFrame(listt)
    df.columns = ['0','1','2','3']
    df.rename(columns = {'0':'Team','1':'Avg Total','2':'Avg Off','3':'Avg Def'},inplace=True)
    return df

#SECTION: DISPLAYING DATA ON THE SITE. 

#Player stats
if selected_page=='Player Stats':
    st.header('Standard Stats from Basketball Reference')
    st.write('Data Dimension: ' + str(df1.shape[0]) + ' rows and ' + str(df1.shape[1]) + ' columns.')
    if scenario=='Regular Season':
        st.dataframe(df1)
    if scenario=='Playoffs':
        st.dataframe(dfP)
    st.header('The Model Data')
#     length = len(df2['Player'])
#     tooltips_df = pd.DataFrame({'Scenario':['']*10,'Year':'','Player':'','Position':'','Age':'','Team':'','G':'Games Played','GS':'Games Started','MP':'Minutes Played','Scoring':'','Passing':'','Rebounds':'','Total Offense':'','Total Defense':'','Total Score':''})
#     df2 = df2.style.set_tooltips(tooltips_df)
    st.dataframe(df2)
    #st.dataframe(df2)

    #Search by player
    st.header('Search By Player')
    players = st.multiselect('Players',list(df2['Player']))
    st.dataframe(byPlayer())

    #Visualization
    st.header('Visualization of Various Parameters Vs. Total Score')
    X = st.selectbox('X Axis',['Age','Team','Total Minutes','Scoring','Passing','Rebounds','Total Offense','Total Defense'])
    fig = px.scatter(df2,x=X,y='Total Score',hover_data=['Player'])
    st.plotly_chart(fig)

    # Heatmap. 
    if st.button('Intercorrelation Heatmap'):
        st.header('Intercorrelation Matrix Heatmap')
        df1.to_csv('output.csv',index=False)
        df = pd.read_csv('output.csv')

        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot(f)

#Team stats
if selected_page=='Team Stats':
    st.header('Average Total Score for Teams')
    st.markdown("""
    Weighted by minutes played
    """)

    st.dataframe(byTeam())
    st.dataframe(byTeam_exp())
    st.dataframe(load_data_exp(selected_year))

    #Plot Teams Data
    st.header('Visualization of Various Parameters Vs. Total Score')
    X2 = st.selectbox('X Axis',['Team', 'Avg Total', 'Avg Off', 'Avg Def', 'Avg Passing', 'Avg Scoring', 'Avg Rebounds', 'Avg Age'])
    fig = px.scatter(byTeam(),x=X2,y='Avg Total', labels={'x':'Index','y':'Avg Score'},text='Team',title="Weighted by Minutes Played")
    fig.update_layout(yaxis_title='Avg Total')
    fig.update_layout(xaxis=dict(showticklabels=True))
    if X2=='Team':
        fig.update_layout(xaxis=dict(showticklabels=False))
    fig.update_traces(marker_opacity=0)

    st.plotly_chart(fig)

    # Download NBA player stats data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df1), unsafe_allow_html=True)
