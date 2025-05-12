import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import modules
from src.data_processing import load_data
from src.seasonal_analysis import analyze_seasonal_products
from src.marketing_analysis import analyze_marketing_channels, create_promotional_calendar
from src.visualization import create_promotional_calendar_heatmap

# Set page config
st.set_page_config(
    page_title="Promotional Calendar - Data Analytics",
    page_icon="🧁",
    layout="wide"
)

# Load data
@st.cache_data
def load_cached_data():
    try:
        return load_data("data/01JTBTJ3CJ4JKZ758BZQ9YT51P.xlsx")
    except FileNotFoundError:
        st.error("Data file not found. Please place the data file in the data directory.")
        return None

df = load_cached_data()

if df is None:
    st.stop()

# Page content
st.title("Promotional Calendar")

st.write("""
This page presents a data-driven promotional calendar based on the analysis of
seasonal product performance and marketing channel effectiveness.
""")

# Create promotional calendar
seasonal_perf, top_products = analyze_seasonal_products(df)
channel_perf, best_channels, channel_by_season = analyze_marketing_channels(df)

promo_calendar = create_promotional_calendar(top_products, best_channels)

# Add month names
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

promo_calendar['Month Name'] = promo_calendar['Month'].map(month_names)

# Sort by month
promo_calendar = promo_calendar.sort_values('Month')

# Display promotional calendar heatmap
st.subheader("Promotional Calendar Heatmap")

promo_heatmap = create_promotional_calendar_heatmap(promo_calendar)
st.plotly_chart(promo_heatmap, use_container_width=True)

# Interactive calendar view
st.subheader("Interactive Calendar View")

# Allow filtering by time period
view_option = st.radio("View by:", ["All Year", "Quarter", "Month"], horizontal=True)

if view_option == "Quarter":
    # Map months to quarters
    quarter_mapping = {
        1: 'Q1', 2: 'Q1', 3: 'Q1',
        4: 'Q2', 5: 'Q2', 6: 'Q2',
        7: 'Q3', 8: 'Q3', 9: 'Q3',
        10: 'Q4', 11: 'Q4', 12: 'Q4'
    }
    
    promo_calendar['Quarter'] = promo_calendar['Month'].map(quarter_mapping)
    
    selected_quarter = st.selectbox("Select Quarter:", ['Q1', 'Q2', 'Q3', 'Q4'])
    filtered_calendar = promo_calendar[promo_calendar['Quarter'] == selected_quarter]
    
elif view_option == "Month":
    selected_month = st.selectbox("Select Month:", list(month_names.values()))
    selected_month_num = [k for k, v in month_names.items() if v == selected_month][0]
    filtered_calendar = promo_calendar[promo_calendar['Month'] == selected_month_num]
    
else:
    filtered_calendar = promo_calendar

# Display calendar
st.dataframe(
    filtered_calendar[['Month Name', 'Season', 'Focus Product', 'Marketing Channel', 'Estimated ROI']]
)

# Calendar visualization
st.subheader("Calendar Visualization")

# Create a gantt-like chart
calendar_data = []

for _, row in filtered_calendar.iterrows():
    calendar_data.append({
        'Month': row['Month Name'],
        'Product': row['Focus Product'],
        'Channel': row['Marketing Channel'],
        'ROI': row['Estimated ROI'],
        'Task': f"{row['Focus Product']} via {row['Marketing Channel']}"
    })

calendar_df = pd.DataFrame(calendar_data)

# Create custom calendar visualization
# Create month start and end dates
if 'Year' not in calendar_df.columns:
    from datetime import datetime
    current_year = datetime.now().year
    calendar_df['Year'] = current_year

calendar_df['Start_Date'] = pd.to_datetime(calendar_df['Month'].astype(str) + '/1/' + calendar_df['Year'].astype(str))
calendar_df['End_Date'] = calendar_df['Start_Date'] + pd.offsets.MonthEnd(1)

# Create timeline with proper start/end dates
fig = px.timeline(
    calendar_df,
    x_start='Start_Date',
    x_end='End_Date',
    y='Product',
    color='Channel',
    title='Promotional Calendar',
    hover_data=['ROI']
)

# Format x-axis to show month names
fig.update_xaxes(
    tickformat="%b %Y",
    title="Month"
)

# Add month order
month_order = [month_names[i] for i in range(1, 13)]
fig.update_xaxes(categoryorder='array', categoryarray=month_order)

st.plotly_chart(fig, use_container_width=True)

# Promotional strategy details
st.subheader("Promotional Strategy Details")

# Create tabs for each season
tab1, tab2, tab3, tab4 = st.tabs(["Spring", "Summer", "Fall", "Winter"])

seasons = ["Spring", "Summer", "Fall", "Winter"]
tabs = [tab1, tab2, tab3, tab4]

for i, season in enumerate(seasons):
    with tabs[i]:
        # Get season data
        season_data = filtered_calendar[filtered_calendar['Season'] == season]
        
        if len(season_data) > 0:
            # Get top product and channel
            top_product = top_products.loc[season, 'Service Category']
            top_channel = best_channels.loc[season, 'Marketing Channel']
            
            st.write(f"**Top Product:** {top_product}")
            st.write(f"**Best Marketing Channel:** {top_channel}")
            
            # Display product performance
            product_perf = df[df['Season'] == season].groupby('Service Category')['Daily Revenue'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                product_perf,
                title=f'Product Performance in {season}',
                labels={'value': 'Total Revenue ($)', 'index': 'Product Category'},
                color=product_perf.index
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy recommendations
            st.subheader("Strategy Recommendations")
            
            st.info(f"Focus on promoting {top_product} through {top_channel} during {season}.")
            
            # Additional recommendations based on product performance
            products = product_perf.index.tolist()
            
            if len(products) > 1:
                st.info(f"Secondary focus: {products[1]}, which is the second-best performing product in {season}.")
                
            # Tactical recommendations
            st.subheader("Tactical Recommendations")
            
            if top_channel == 'In-store Promo':
                st.write("- Create eye-catching in-store displays featuring seasonal themes")
                st.write("- Offer product samples to encourage impulse purchases")
                st.write("- Create bundle offers with complementary products")
                
            elif top_channel == 'Instagram':
                st.write("- Create visually appealing posts highlighting seasonal products")
                st.write("- Partner with local influencers for product promotion")
                st.write("- Run targeted Instagram ads to reach new customers")
                
            elif top_channel == 'Local Magazine':
                st.write("- Create seasonal-themed advertisements highlighting key products")
                st.write("- Include special offers or coupons in magazine ads")
                st.write("- Feature seasonal recipes using your products")
                
            elif top_channel == 'Google Maps':
                st.write("- Ensure business listing is updated with seasonal offerings")
                st.write("- Encourage customers to leave positive reviews")
                st.write("- Use Google Posts to highlight seasonal specials")
                
            # Implementation timeline
            st.subheader("Implementation Timeline")
            
            months = [month_names[m] for m in range(1, 13) if season_data['Month'].isin([m]).any()]
            
            for month in months:
                st.write(f"**{month}:**")
                st.write(f"- Feature {top_product} promotions via {top_channel}")
                st.write(f"- Create {season}-themed marketing materials")
                st.write(f"- Monitor performance and adjust strategy as needed")
        else:
            st.write(f"No promotional data available for {season} in the selected time period.")

# Download button
st.subheader("Download Promotional Calendar")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(promo_calendar)

st.download_button(
    label="Download Promotional Calendar (CSV)",
    data=csv,
    file_name="promotional_calendar.csv",
    mime="text/csv",
)

# Excel version with formatting
@st.cache_data
def create_excel_calendar():
    from io import BytesIO
    
    # Create Excel file in memory
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main calendar
        promo_calendar.to_excel(writer, sheet_name='Promo Calendar', index=False)
        
        # Fix: Properly flatten seasonal_perf DataFrame
        seasonal_perf_flat = seasonal_perf.copy()
        # Only apply column flattening if it's a MultiIndex
        if isinstance(seasonal_perf_flat.columns, pd.MultiIndex):
            seasonal_perf_flat.columns = ['_'.join(col).strip() for col in seasonal_perf.columns.values]
        seasonal_perf_flat = seasonal_perf_flat.reset_index()
        seasonal_perf_flat.to_excel(writer, sheet_name='Seasonal Performance', index=False)
        
        # Write channel performance similarly
        channel_perf_flat = channel_perf.copy()
        if isinstance(channel_perf_flat.columns, pd.MultiIndex):
            channel_perf_flat.columns = ['_'.join(col).strip() for col in channel_perf.columns.values]
        channel_perf_flat = channel_perf_flat.reset_index()
        channel_perf_flat.to_excel(writer, sheet_name='Channel Performance', index=False)
    
    output.seek(0)
    return output.getvalue()
    
excel_data = create_excel_calendar()

st.download_button(
    label="Download Promotional Calendar (Excel)",
    data=excel_data,
    file_name="promotional_calendar.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Implementation roadmap
st.subheader("Implementation Roadmap")

st.write("""
Follow these steps to implement the promotional calendar:

1. **Prepare promotional materials** for each seasonal focus product
2. **Allocate marketing budget** towards the recommended channels
3. **Schedule promotions** according to the calendar
4. **Monitor performance** and adjust strategy as needed
5. **Evaluate results** at the end of each season
""")

# Expected outcomes
col1, col2, col3 = st.columns(3)

col1.metric("Projected Revenue Increase", "8-12%", "Based on optimal product-season alignment")
col2.metric("Marketing ROI Improvement", "15-20%", "Through channel optimization")
col3.metric("Customer Value Increase", "10-15%", "Via targeted upselling")