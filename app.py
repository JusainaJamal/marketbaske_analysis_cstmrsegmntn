import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from st_aggrid import AgGrid

st.set_page_config(layout="wide")

def main2():
    st.title("Retailers App")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state.logged_in:
        show_login_page()
    else:
        app()

def show_login_page():
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "your_username" and password == "your_password":
            st.session_state.logged_in = True
            st.experimental_rerun()  # Rerun the app to redirect to welcome page
        else:
            st.error("Invalid username or password")


def app():
    st.subheader("Welcome")
    st.write("You are logged in! Welcome to our application.")

    # Set page config
    # st.set_page_config(layout="wide")
    page = st.sidebar.radio("Navigation Pane:", ["Product Recommendation using Market Basket Analysis", "Customer Segmentation based on RFM Analysis", "Dashboard","Predict Customer Cluster"])
   
    # Add sidebar to the app
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")

    # Read clean dataset

    retail = pd.read_csv('.ipynb_checkpoints/CleanRetailData.csv') 
    
    # List of all countries in dataset
    country_list = list(dict(retail['Country'].value_counts()).keys())
    
    # Subsetting retail dataframe based on country
    def choose_country(country = "all", data = retail):
        if country == "all":
            return data
        else:
            temp_df = data[data["Country"] == country]
            temp_df.reset_index(drop= True, inplace= True)
            return temp_df
             
    # For United Kingdom, since it contains majority of data
    uk_retail = choose_country("United Kingdom")
        
    def cluster_plot(data_frame):
        fig = px.scatter_3d(data_frame, x = 'Recency', y='Frequency', z='Monetary',
                  color='Clusters', opacity = 0.8, width=600, height=600, template="plotly_dark")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, height=600)
        
    def kmeans_on_df():
        # Scaling Recency, Frequency, Monetary and RFM_Score columns
        scaler = StandardScaler()
        # Subset
        rfm_scaled = rfm_new[['Recency','Frequency','Monetary','RFMScore']]
        rfm_scaled = scaler.fit_transform(rfm_scaled)
        rfm_scaled = pd.DataFrame(rfm_scaled, columns = ['Recency','Frequency','Monetary','RFMScore'])
        
        # Fit Kmeans at n_clusters = 4
        kmeans = KMeans(n_clusters=4, init='k-means++',n_init=10,max_iter=50,verbose=0)
        kmeans.fit(rfm_scaled)
        
        # Assigning Clusters
        rfm_new['Clusters'] = kmeans.labels_
        
        return rfm_new
    
    def plot_pcts(df, string):
        # https://sakizo-blog.com/en/607/
        fig_target = go.Figure(data=[go.Pie(labels=df.index,
                                    values=df[string],
                                    hole=.3)])
        fig_target.update_layout(showlegend=False,
                                 height=500,
                                 margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
        fig_target.update_traces(textposition='inside', textinfo='label+percent')
        fig_target.update_traces(marker=dict(colors=['lightcyan', 'cyan', 'royalblue', 'darkblue']))
        return st.plotly_chart(fig_target, use_container_width=True)
    
    # Function to group on Month/Date/Day of the Week/Week of the Year/Time of the Day
    def group_sales_quantity(df, feature):
        df = df[[f'{feature}','Quantity','Sales Revenue']].groupby([f'{feature}']).sum().sort_values(by= 'Sales Revenue', ascending = False).reset_index()
        return df
# %%    
    # First page
    if page == "Product Recommendation using Market Basket Analysis":
        # Title
        html_temp_title = """
            <div style="background-color:#154360;padding:2px">
            <h2 style="color:white;text-align:center;">Product Recommendation using Market Basket Analysis</h2>
            </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("")

        # Pick country 
        st.markdown('### Choose Country:')
        option = st.selectbox('', country_list)
        country_retail = choose_country(option)
        
        # Display
        AgGrid(country_retail, theme='streamlit', height = 200, width = 150)
                    
        # List of all products
        product_catalog = list(country_retail['Description'].unique())
        
        #### Need to have a drop down to choose country and then filter dataset based on that
        st.markdown('### Choose Product:')
        prod_option = st.selectbox('', product_catalog)
        
        # Opening JSON file
        with open('item_sets.json') as json_file:
            data = json.load(json_file)
       
        # Display    
        if len(data[prod_option]) == 0:
            st.error("Oops! No product recommendations available yet! Please select a different item.")
        else:
            st.markdown("####")
            st.success("##### People also bought...")
            for d in data[prod_option]:
                if d:
                    st.markdown("- " + d)
# %%
    if page == "Customer Segmentation based on RFM Analysis":
        # Title
        html_temp_title = """
            <div style="background-color:#154360;padding:2px">
            <h2 style="color:white;text-align:center;">Customer Segmentation based on RFM Analysis</h2>
            </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("")

        try:                   
            col1, col2, col3= st.columns([5, 1, 10])
        
            with col1:
                # About RFM
                st.markdown('## .')
                
            with col3:
                # Pick country 
                st.markdown('## Choose Country:')
                rfm_country = st.selectbox('', country_list)
                rfm_country_df = choose_country(rfm_country)
               
                # We need a reference day to perform the RFM Analysis
                # In this case the day after the last recorded date in the dataset plus a day
                rfm_country_df['InvoiceDate'] = pd.to_datetime(rfm_country_df['InvoiceDate'])
                ref_date = rfm_country_df['InvoiceDate'].max() + dt.timedelta(days=1)
                
                # Remove 'Guest Customer' 
                rfm_country_df = rfm_country_df[rfm_country_df['CustomerID'] != "Guest Customer"]
                
                # Aggregating over CustomerID
                rfm_new = rfm_country_df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (ref_date - x.max()).days,
                                            'InvoiceNo': lambda x: x.nunique(),
                                            'Sales Revenue': lambda x: x.sum()})
                # Calculate quantiles
                rfm_new.columns = ['Recency', 'Frequency', 'Monetary']
                rfm_new["R"] = pd.qcut(rfm_new['Recency'].rank(method="first"), 4, labels=[4, 3, 2, 1])
                rfm_new["F"] = pd.qcut(rfm_new['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
                rfm_new["M"] = pd.qcut(rfm_new['Monetary'].rank(method="first"), 4, labels=[1, 2, 3, 4])
                
                # Calculate RFM Score
                rfm_new['RFMScore'] = (rfm_new['R'].astype(int) + rfm_new['F'].astype(int) + rfm_new['M'].astype(int))
                
                # New RFM Dataframe
                rfm_new.reset_index(inplace=True)
                
                # K-means
                df = kmeans_on_df()
                
                # Display merged dataframes   
                AgGrid(df,  theme='streamlit', height=200, width=150)
            
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">Identified Clusters</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            cluster_plot(df)
            
            # Pie charts
            rfm_stats = df[["Clusters","RFMScore", "Recency", "Frequency", "Monetary"]].groupby("Clusters").agg(["mean"])
            rfm_stats.columns = ["RFM_Score_Mean", "Recency_Mean", "Frequency_Mean", "Monetary_Mean"]
            
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">Pie-plot Distribution of Clusters Based on RFM Analysis</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown('###')
            col1, col2, col3, col4, col5 = st.columns([5, 1, 5, 1, 5])
            with col1:
                html_temp_title = """
                <div style="background-color:lightblue;padding:4px">
                <h5 style="color:white;text-align:center;">Recency</h5>
                </div>
                """
                st.markdown(html_temp_title, unsafe_allow_html=True)
                plot_pcts(rfm_stats, 'Recency_Mean')
            with col3:
                html_temp_title = """
                <div style="background-color:lightblue;padding:4px">
                <h5 style="color:white;text-align:center;">Frequency</h5>
                </div>
                """
                st.markdown(html_temp_title, unsafe_allow_html=True)
                plot_pcts(rfm_stats, 'Frequency_Mean')
            with col5:
                html_temp_title = """
                <div style="background-color:lightblue;padding:4px">
                <h5 style="color:white;text-align:center;">Monetary</h5>
                </div>
                """
                st.markdown(html_temp_title, unsafe_allow_html=True)
                plot_pcts(rfm_stats, 'Monetary_Mean')
            
        except:
            st.error("Oops! Error performing operation! Please select another country.")
# %%
    if page == "Dashboard":
        # Title
        html_temp_title = """
            <div style="background-color:#154360;padding:2px">
            <h2 style="color:white;text-align:center;">Dashboard</h2>
            </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        
        # Pick country 
        st.markdown('#### Choose Country:')
        option = st.selectbox('', country_list)
        country_retail = choose_country(option)
        
        # Top 10 customers without 'Guest Customer'
        top_customers = country_retail[country_retail["CustomerID"] != "Guest Customer"].groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(11)
        html_temp_title = """
        <div style="background-color:#ABBAEA;padding:4px">
        <h3 style="color:white;text-align:center;">Top Customers without 'Guest Customer'</h3>
        </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        fig = px.bar(top_customers, x ="CustomerID", y = "InvoiceNo", color= 'InvoiceNo')
        fig.update_layout(showlegend=False,
                                 height=250, width = 500,
                                 margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
        fig.update(layout_coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Top 10 performing products 
        col1, col2, col3= st.columns([10, 1, 10])
        with col1:
            html_temp_title = """
            <div style="background-color:#ABBAEA;padding:4px">
            <h3 style="color:white;text-align:center;">Top Products by Sold Quantity</h3>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            top_products_qty = group_sales_quantity(country_retail, 'Description').sort_values(ascending=False, by = "Quantity").reset_index(drop=True)
            top_products_qty.drop('Sales Revenue', axis=1, inplace=True)
            # Display merged dataframes   
            AgGrid(top_products_qty,theme='streamlit', height=200, width=150)
        with col3:
            html_temp_title = """
            <div style="background-color:#ABBAEA;padding:4px">
            <h3 style="color:white;text-align:center;">Top Products by Gross Sales Revenue</h3>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            # Top 10 Product Description by Sales Revenue
            top_products_revenue = group_sales_quantity(country_retail, 'Description').sort_values(ascending=False, by = "Sales Revenue").reset_index(drop=True)
            top_products_revenue.drop('Quantity', axis=1, inplace=True)
            # Display merged dataframes   
            AgGrid(top_products_revenue, theme='streamlit', height=200, width=150)
# %%
    if page == "Predict Customer Cluster":
        # Title
        html_temp_title = """
            <div style="background-color:#154360;padding:2px">
            <h2 style="color:white;text-align:center;">Predict Customer Cluster</h2>
            </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        
        # Pick country 
        st.markdown('### Choose Country:')
        option = st.selectbox('', country_list)
        country_retail = choose_country(option)
        # Clusters for selected country
        rfm_country_df = choose_country(option)
        
        # We need a reference day to perform the RFM Analysis
        # In this case the day after the last recorded date in the dataset plus a day
        rfm_country_df['InvoiceDate'] = pd.to_datetime(rfm_country_df['InvoiceDate'])
        ref_date = rfm_country_df['InvoiceDate'].max() + dt.timedelta(days=1)
        
        # Remove 'Guest Customer' 
        rfm_country_df = rfm_country_df[rfm_country_df['CustomerID'] != "Guest Customer"]
        
        # Aggregating over CustomerID
        rfm_new = rfm_country_df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (ref_date - x.max()).days,
                                    'InvoiceNo': lambda x: x.nunique(),
                                    'Sales Revenue': lambda x: x.sum()})
        # Calculate quantiles
        rfm_new.columns = ['Recency', 'Frequency', 'Monetary']
        rfm_new["R"] = pd.qcut(rfm_new['Recency'].rank(method="first"), 4, labels=[4, 3, 2, 1])
        rfm_new["F"] = pd.qcut(rfm_new['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
        rfm_new["M"] = pd.qcut(rfm_new['Monetary'].rank(method="first"), 4, labels=[1, 2, 3, 4])
        
        # Calculate RFM Score
        rfm_new['RFMScore'] = (rfm_new['R'].astype(int) + rfm_new['F'].astype(int) + rfm_new['M'].astype(int))
        
        # New RFM Dataframe
        rfm_new.reset_index(inplace=True)
        
        # K-means
        df = kmeans_on_df()
        
        # Display merged dataframes   
        AgGrid(df,  theme='streamlit', height=200, width=150)
        
        # Clustering 
        html_temp_title = """
        <div style="background-color:SteelBlue;padding:4px">
        <h4 style="color:white;text-align:center;">Identified Clusters</h4>
        </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        cluster_plot(df)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()  # Rerun the app to redirect to login page
    

# Execute the main function
if __name__ == "__main__":
    main2() 
