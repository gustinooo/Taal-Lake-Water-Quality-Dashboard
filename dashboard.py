import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
csv_file = "Group 3 Water Quality Dataset - FINAL DATASET.csv"
df = pd.read_csv(csv_file)

# Data preprocessing
df.replace('ND', pd.NA, inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(subset=['Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
                  'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen', 'Sulfide',
                  'Carbon Dioxide', 'Air Temperature'], inplace=True)

month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month'] = df['Month '].map(month_mapping)
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df["Weather Condition"] = df["Weather Condition"].str.strip()
df["Weather Condition Code"], uniques = pd.factorize(df["Weather Condition"])
weather_mapping = dict(enumerate(uniques))

wind_direction_cols = [col for col in df.columns if 'Wind Direction' in col]
wind_direction_mappings = {}
for col in wind_direction_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col + '_Code'], uniques = pd.factorize(df[col])
    wind_direction_mappings[col] = dict(enumerate(uniques))

numeric_cols = ['Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
                'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
                'Air Temperature', 'Sulfide', 'Carbon Dioxide']

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Set up the page layout
st.set_page_config(page_title="Taal Lake Water Quality Dashboard", page_icon="📊", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Exploratory Data Analysis for Trends", "Exploratory Data Analysis for each Parameter", "Heatmap of Average Parameters", "Correlation Matrix", "Scatter Plots for Parameter Pairs", "Deep Learning Model for Water Temperature Predictions", "Deep Learning Model for Water Temperature + Water Quality Predictions", "Deep Learning Model for Water Temperature + Water Quality Predictions + Volcanic Activity Predictions", "Deep Learning Model for Water Temperature + Water Quality Predictions + Volcanic Activity + Weather Predictions"])

# Page 1: Dashboard
if page == "Dashboard":
    # Create two columns
    col1, col2 = st.columns([1, 2]) 

    # Column 1: Details
    with col1:
        st.title(":bar_chart::volcano: Taal Lake Water Quality Data Analysis Dashboard")
        st.markdown("This dashboard provides an overview of the Taal Lake Water Quality dataset and its key statistics. :sparkles:")
        st.markdown("Presented by: Group 3")
        st.markdown("**Group Members:**\n- Augustine Sengodayan\n- Yannah Gonzales\n- Vladymir Bisares\n- Blessie Faith Gomez\n- Krysschell Anne Andalahaw")

    # Column 2: Data Viewer
    with col2:
        st.title("Taal Lake Water Quality Dataset")
        st.write(df)
        st.markdown("**Note:** The dataset contains various water temperature, water quality, volcanic activity, and weather condition parameters.")


elif page == "Exploratory Data Analysis for Trends": 
    st.title("Exploratory Data Analysis for Trends")

    # Select only numeric columns for averaging
    numeric_columns = df.select_dtypes(include=["number"]).columns
    averaged_data = df.groupby("Date")[numeric_columns].mean().reset_index()

    # Water Temperature Trends
    st.markdown("### Water Temperature Trends")
    st.markdown(
        "Water temperature plays a crucial role in aquatic ecosystems, influencing both biological and chemical processes. "
        "This visualization tracks the average surface, middle, and bottom temperatures over time, providing valuable insights into "
        "seasonal variations and potential environmental shifts."
    )
    fig = px.line(averaged_data, x="Date", y=["Surface Temperature", "Middle Temperature", "Bottom Temperature"])
    st.plotly_chart(fig)

    # Water Quality Trends
    st.markdown("### Water Quality Trends")
    st.markdown(
        "Water quality is essential for sustaining life. This chart illustrates the average variations in key indicators such as "
        "pH, ammonia, nitrate, phosphate, and dissolved oxygen levels. Monitoring these trends allows researchers and "
        "environmentalists to assess pollution levels, water safety, and overall ecosystem stability."
    )
    fig = px.line(averaged_data, x="Date", y=["pH", "Ammonia", "Nitrate", "Phosphate", "Dissolved Oxygen"])
    st.plotly_chart(fig)

    # Volcanic Activity Trends
    st.markdown("### Volcanic Activity Trends")
    st.markdown(
        "Volcanic activity can have a significant impact on water chemistry, affecting both aquatic organisms and local environments. "
        "This visualization showcases the average fluctuation of sulfide and carbon dioxide concentrations in water, helping to identify "
        "possible correlations with volcanic activity."
    )
    fig = px.line(averaged_data, x="Date", y=["Sulfide", "Carbon Dioxide"])
    st.plotly_chart(fig)

    # Weather Condition Trends
    st.markdown("### Weather Condition Trends")
    st.markdown(
        "Weather patterns directly affect water quality and ecological balance. This chart examines the average air temperature, weather conditions, "
        "and wind direction, revealing how atmospheric changes may influence aquatic ecosystems."
    )
    fig = px.line(averaged_data, x="Date", y=['Air Temperature', 'Weather Condition Code', 'Wind Direction_Code'])
    st.plotly_chart(fig)

# Page 3: Visualizations for Each Parameter
elif page == "Exploratory Data Analysis for each Parameter":  
    st.title("Exploratory Data Analysis for each Parameter")
    # List of parameters to visualize
    parameters = [
        "Surface Temperature", "Middle Temperature", "Bottom Temperature",
        "pH", "Ammonia", "Nitrate", "Phosphate", "Dissolved Oxygen",
        "Sulfide", "Carbon Dioxide", "Air Temperature"
    ]

    # Dictionary to store descriptions for each parameter
    parameter_descriptions = {
        "Surface Temperature": "Surface temperature is a critical factor in aquatic ecosystems, influencing the behavior and survival of aquatic organisms.",
        "Middle Temperature": "Middle temperature provides insights into the thermal stratification of the water column, which affects nutrient cycling.",
        "Bottom Temperature": "Bottom temperature trends help monitor the conditions at the lakebed, which can impact benthic organisms.",
        "pH": "pH levels indicate the acidity or alkalinity of the water, which is vital for maintaining aquatic life.",
        "Ammonia": "Ammonia levels are an indicator of pollution and can be toxic to aquatic organisms at high concentrations.",
        "Nitrate": "Nitrate levels are essential for understanding nutrient loading and potential eutrophication in the lake.",
        "Phosphate": "Phosphate levels are a key factor in assessing the risk of algal blooms and water quality degradation.",
        "Dissolved Oxygen": "Dissolved oxygen is crucial for the survival of aquatic organisms and reflects the overall health of the ecosystem.",
        "Sulfide": "Sulfide concentrations can indicate volcanic activity and its impact on water chemistry.",
        "Carbon Dioxide": "Carbon dioxide levels are important for understanding the lake's carbon cycle and its interactions with the atmosphere.",
        "Air Temperature": "Air temperature trends provide context for understanding how atmospheric conditions influence the lake's ecosystem."
    }

    # Loop through parameters and generate visualizations with descriptions
    for parameter in parameters:
        st.markdown(f"#### {parameter} Trends Across Sites")
        st.markdown(parameter_descriptions.get(parameter, "No description available for this parameter."))
        fig = px.line(
            df, x="Date", y=parameter, color="Site",
        )
        st.plotly_chart(fig)

# Page 4: Heatmap of Average Parameters Over Time
elif page == "Heatmap of Average Parameters":
    st.title("Heatmap of Average Temperature and Water Quality Parameters Over Time")

    # Description of the heatmap
    st.markdown(
        "This heatmap provides a visual representation of the average values of temperature and water quality parameters over time. "
        "Each row represents a specific parameter, while each column corresponds to a date. "
        "The color intensity indicates the average value of the parameter, with darker colors representing higher values. "
    )
    # Select relevant columns for the heatmap
    heatmap_columns = [
        "Surface Temperature", "Middle Temperature", "Bottom Temperature",
        "pH", "Ammonia", "Nitrate", "Phosphate", "Dissolved Oxygen"
    ]

    # Group data by Date and calculate the mean for each parameter
    heatmap_data = df.groupby("Date")[heatmap_columns].mean()

    # Create the heatmap using Plotly
    fig = px.imshow(
        heatmap_data.T,  # Transpose to have parameters as rows and dates as columns
        labels=dict(x="Date", y="Parameter", color="Average Value"),
        x=heatmap_data.index,
        y=heatmap_columns,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)  # Make the chart span across the page

# Page 5: Correlation Matrix
elif page == "Correlation Matrix":
    st.title("Correlation Matrix of Temperature, Water Quality, and Other Parameters")

    # Description of the correlation matrix
    st.markdown(
        "This correlation matrix shows the relationships between temperature, water quality, and other parameters. "
        "The values range from -1 to 1, where:\n"
        "- **1** indicates a perfect positive correlation.\n"
        "- **-1** indicates a perfect negative correlation.\n"
        "- **0** indicates no correlation.\n\n"
        "This visualization helps identify strong relationships between parameters, which can provide insights into the ecosystem's dynamics."
    )

    # Select relevant columns for the correlation matrix
    correlation_columns = [
        "Surface Temperature", "Middle Temperature", "Bottom Temperature",
        "pH", "Ammonia", "Nitrate", "Phosphate", "Dissolved Oxygen",
        "Sulfide", "Carbon Dioxide", "Air Temperature"
    ]

    # Compute the correlation matrix
    correlation_matrix = df[correlation_columns].corr()

    # Create the heatmap using Plotly
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Parameters", y="Parameters", color="Correlation"),
        x=correlation_columns,
        y=correlation_columns,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation Matrix of Parameters"
    )

    # Adjust the figure size to span the full width
    fig.update_layout(
        autosize=True,
        width=1200,  # Set a custom width
        height=800   # Set a custom height
    )

    # Render the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Page 6: Scatter Plots for Parameter Pairs
elif page == "Scatter Plots for Parameter Pairs":
    st.title("Scatter Plots for Parameter Pairs")

    # Description of the scatter plots
    st.markdown(
        "This page displays scatter plots for selected parameter pairs to visualize their relationships. "
        "Scatter plots help identify patterns, trends, and potential correlations between two variables."
    )

    # List of parameter pairs to visualize
    parameter_pairs = [
        ('Surface Temperature', 'Middle Temperature'),
        ('Surface Temperature', 'Bottom Temperature'),
        ('Dissolved Oxygen', 'pH'),
        ('Ammonia', 'Nitrate'),
        ('Ammonia', 'Phosphate'),
        ('Nitrate', 'Phosphate'),
        ('Surface Temperature', 'Dissolved Oxygen'),
        ('Surface Temperature', 'Ammonia'),
        ('Surface Temperature', 'Nitrate'),
        ('Middle Temperature', 'pH'),
        ('Bottom Temperature', 'pH'),
        ('Weather Condition Code', 'Surface Temperature'),
        ('Wind Direction_Code', 'Surface Temperature'),
        ('Weather Condition Code', 'Nitrate')
    ]

    # Loop through each parameter pair and generate scatter plots
    for x_param, y_param in parameter_pairs:
        st.markdown(f"#### Scatter Plot: {x_param} vs {y_param}")
        fig = px.scatter(
            df, x=x_param, y=y_param, color="Site",
            title=f"{x_param} vs {y_param}",
            labels={x_param: x_param, y_param: y_param},
            template="plotly"
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 7: Deep Learning Model for Water Temperature Predictions
elif page == "Deep Learning Model for Water Temperature Predictions":
    st.markdown("### CNN, LSTM, and Hybrid Model for Water Temperature Prediction")


    # Define model performance data
    data = {
        "Model": ["CNN", "LSTM", "Hybrid CNN-LSTM"],
        "MAE": [0.1250, 0.1333, 0.1309],
        "RMSE": [0.1483, 0.1559, 0.1550],
        "R²": [0.6643, 0.6289, 0.6331]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    st.write("Below is the table showing model performance metrics:")

    # Display table
    st.dataframe(df, hide_index=True)
    
    # Insert an image at the bottom
    st.image(r"Water Temp 1 Predictions 1.png", caption="Actual vs. Predicted Values", use_container_width=True)

    col1, col2 = st.columns([1, 2]) 

    # Column 1: Details
    with col1:
        st.image(r"Water Temp 1 Predictions 2.png", caption="Model Performance Comparisson", use_container_width=True)
    with col2:
        st.image(r"Water Temp 1 Predictions 3.png", caption="Distribution Table for Water Quality Index", use_container_width=True)

# Page 8: Deep Learning Model for Predictions
elif page == "Deep Learning Model for Water Temperature + Water Quality Predictions":
    st.markdown("### CNN, LSTM, and Hybrid Model for Water Temperature + Water Quality Prediction")

    # Define model performance data
    data = {
        "Model": ["CNN", "LSTM", "Hybrid CNN-LSTM"],
        "MAE": [0.1393, 0.1378, 0.1326],
        "RMSE": [0.1600, 0.1567, 0.1525],
        "R²": [0.6095, 0.6252, 0.6451]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    st.write("Below is the table showing updated model performance metrics:")

    # Display table
    st.dataframe(df, hide_index=True)

    
    # Insert an image at the bottom
    st.image(r"Water Temp 2 Predictions 1.png", caption="Actual vs. Predicted Values", use_container_width=True)

    col1, col2 = st.columns([1, 2]) 

    # Column 1: Details
    with col1:
        st.image(r"Water Temp 2 Predictions 2.png", caption="Model Performance Comparisson", use_container_width=True)
    with col2:
        st.image(r"Water Temp 2 Predictions 3.png", caption="Distribution Table for Water Quality Index", use_container_width=True)
    
    st.image(r"Water Temp 2 Predictions 4.png", caption="Average Pollutant Levels per Date", use_container_width=True)

# Page 9: Deep Learning Model for Predictions
elif page == "Deep Learning Model for Water Temperature + Water Quality Predictions + Volcanic Activity Predictions":
    st.markdown("### CNN, LSTM, and Hybrid Model for Water Temperature + Water Quality + Volcanic Activity Prediction")

    # Define model performance data
    data = {
    "Model": ["CNN", "LSTM", "Hybrid CNN-LSTM"],
    "MAE": [0.1196, 0.1267, 0.1149],
    "RMSE": [0.1478, 0.1490, 0.1459],
    "R²": [0.6665, 0.6611, 0.6751]
}

    # Create a DataFrame
    df = pd.DataFrame(data)

    st.write("Below is the table showing updated model performance metrics:")

    # Display table
    st.dataframe(df, hide_index=True)


    
    # Insert an image at the bottom
    st.image(r"Water Temp 3 Predictions 1.png", caption="Actual vs. Predicted Values", use_container_width=True)

    col1, col2 = st.columns([1, 2]) 

    # Column 1: Details
    with col1:
        st.image(r"Water Temp 3 Predictions 2.png", caption="Model Performance Comparisson", use_container_width=True)
    with col2:
        st.image(r"Water Temp 3 Predictions 3.png", caption="Distribution Table for Water Quality Index", use_container_width=True)
    
    st.image(r"Water Temp 3 Predictions 4.png", caption="Average Pollutant Levels per Date", use_container_width=True)

# Page 10: Deep Learning Model for Predictions
elif page == "Deep Learning Model for Water Temperature + Water Quality Predictions + Volcanic Activity + Weather Predictions":
    st.markdown("### CNN, LSTM, and Hybrid Model for Water Temperature + Water Quality + Volcanic Activity + Weather Prediction")

    # Define model performance data
    data = {
        "Model": ["CNN", "LSTM", "Hybrid CNN-LSTM"],
        "MAE": [0.1860, 0.1344, 0.1590],
        "RMSE": [0.2252, 0.1682, 0.1938],
        "R²": [0.2258, 0.5680, 0.4266]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    st.write("Below is the table showing updated model performance metrics:")

    # Display table
    st.dataframe(df, hide_index=True)


    
    # Insert an image at the bottom
    st.image(r"Water Temp 4 Predictions 1.png", caption="Actual vs. Predicted Values", use_container_width=True)

    col1, col2 = st.columns([1, 2]) 

    # Column 1: Details
    with col1:
        st.image(r"Water Temp 4 Predictions 2.png", caption="Model Performance Comparisson", use_container_width=True)
    with col2:
        st.image(r"Water Temp 4 Predictions 3.png", caption="Distribution Table for Water Quality Index", use_container_width=True)
    
    st.image(r"Water Temp 4 Predictions 4.png", caption="Average Pollutant Levels per Date", use_container_width=True)
