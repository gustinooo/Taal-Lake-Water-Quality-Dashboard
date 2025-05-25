import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


# Load the CSV file
csv_file = "C:/Users/August/Documents/CPEN 106/MC02/Group 3 Water Quality Dataset - FINAL DATASET.csv"
df = pd.read_csv(csv_file)

# Data preprocessing
df.replace('ND', pd.NA, inplace=True)
df.drop_duplicates(inplace=True)
columns_to_fill = ['Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
                   'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen', 
                   'Sulfide', 'Carbon Dioxide', 'Air Temperature']
for col in columns_to_fill:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric safely
    df[col].fillna(df[col].mean(), inplace=True)  # Replace NaNs with column mean

# Date and mapping adjustments
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month'] = df['Month '].map(month_mapping)
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df["Weather Condition"] = df["Weather Condition"].str.strip()
df["Weather Condition Code"], uniques = pd.factorize(df["Weather Condition"])
weather_mapping = dict(enumerate(uniques))

# Encode wind directions
wind_direction_cols = [col for col in df.columns if 'Wind Direction' in col]
wind_direction_mappings = {}
for col in wind_direction_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col + '_Code'], uniques = pd.factorize(df[col])
    wind_direction_mappings[col] = dict(enumerate(uniques))

# Normalize numeric columns
numeric_cols = ['Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
                'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
                'Air Temperature', 'Sulfide', 'Carbon Dioxide']

scaler = MinMaxScaler()
for col in numeric_cols:
    df[f"{col}_Normalized"] = scaler.fit_transform(df[[col]])



def page1():
    # Create two columns
    col1, col2 = st.columns([4, 2])  # Adjust column widths as needed
    # Column 1: Objectives
    with col1:
        st.markdown("### Overview")
        st.markdown("""
        This lab aims to apply data mining and data visualization techniques to predict water quality in Taal Lake using real-world datasets. 
        Students will collect, preprocess, analyze, and model data to predict parameters such as pH and dissolved oxygen levels. 
        The project incorporates machine learning models and interactive visualizations.
        """)

        st.markdown("### Objectives")
        st.markdown("""
        - Apply data mining techniques to extract useful insights from environmental datasets.
        - Develop predictive models for water quality using machine learning.
        - Compare ensemble learning techniques such as CNN, LSTM, and Hybrid CNN-LSTM.
        - Visualize trends and patterns in water quality parameters.
        - Interpret the impact of environmental and volcanic activity on water quality.
        - Predict Water Quality Index (WQI) and Water Pollutant Levels with actionable insights for environmental monitoring and intervention.
        """)

    # Column 2: Photo
    with col2:
        st.image("C:/Users/August/Documents/CPEN 106/MC02/taal_lake.jpg", caption="Taal Lake", use_container_width=True)

    # Introduce the dataset
    st.markdown("This dashboard provides an overview of the Taal Lake Water Quality dataset and its key statistics.")
    st.write(df)
    st.markdown("**Note:** The dataset contains various water temperature, water quality, volcanic activity, and weather condition parameters.")


def page2():
    # User selects parameters based on actual values
    selected_params = st.multiselect("Select parameters to analyze (Actual Values):", [
        'Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
        'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
        'Air Temperature', 'Sulfide', 'Carbon Dioxide'
    ], default=['Surface Temperature', 'pH', 'Dissolved Oxygen'])

    # Map selections to normalized columns for visualization
    normalized_params = [f"{param}_Normalized" for param in selected_params]

    if selected_params:
        # Merge actual values into a separate dataframe for hover tooltips
        df_display = df.copy()
        for param in selected_params:
            df_display[f"{param}_Actual"] = df[param]  # Retain actual values for interaction

        # --- LINE GRAPH: Normalized Values, Hover Shows Actual Values ---
        fig_line = px.line(df_display, x="Date", y=normalized_params,
                           title="Comparison of Selected Parameters Over Time (Normalized)",
                           labels={"value": "Normalized Value", "variable": "Parameter"},
                           hover_data={param: True for param in selected_params})  # Show actual values on hover

        fig_line.update_layout(legend_title="Parameters", autosize=True)
        st.plotly_chart(fig_line, use_container_width=True)

        col1, col2 = st.columns(2)  # Split layout into two columns

        with col1:
            corr_matrix = df[normalized_params].corr().round(2)

            fig_corr = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                annotation_text=np.array(corr_matrix).astype(str),
                colorscale="viridis"
            )

            fig_corr.update_layout(title="Correlation Matrix (Normalized Values)", 
                                   xaxis=dict(title="Features"), 
                                   yaxis=dict(title="Features"), autosize=True)

            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            # Map selected parameters to their normalized counterparts
            normalized_params = [f"{param}_Normalized" for param in selected_params]

            # Group data by Month and Date using normalized values
            df_grouped = df.groupby([df['Date'].dt.strftime('%Y-%m')])[normalized_params].mean()

            # Create the heatmap using normalized values
            fig_heatmap = px.imshow(
                df_grouped.T,  # Transpose to have parameters as rows and months as columns
                labels=dict(x="Month", y="Features", color="Normalized Value"),
                title="Heatmap of Selected Parameters (Normalized, Grouped by Month)",
                color_continuous_scale="viridis"
            )

            fig_heatmap.update_xaxes(side="bottom")  # Ensure x-axis labels are at the bottom
            st.plotly_chart(fig_heatmap, use_container_width=True)

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(1)
    ])
    return model

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Flatten(),
        Dense(1)
    ])
    return model

def build_hybrid_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        LSTM(50, return_sequences=True),
        Flatten(),
        Dense(1)
    ])
    return model


# --- Page 3: Water Quality Prediction ---
def page3():
    col1, col2 = st.columns(2)
    with col1:
        # Step 1: Choose a target variable
        target_variable = st.selectbox("Select Target Variable:", ['Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
            'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
            'Air Temperature', 'Sulfide', 'Carbon Dioxide', 'Weather Condition Code'])

        # Step 2: Select features
        feature_options = [
            'Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
            'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
            'Air Temperature', 'Sulfide', 'Carbon Dioxide', 'Weather Condition Code',
        ]
        features = st.multiselect("Select Features:", feature_options, default=['Surface Temperature', 'Middle Temperature', 'Bottom Temperature'])

        if target_variable and features:
            # Ensure the index is reset for proper selection
            df.reset_index(inplace=True)

            # Prepare dataset
            X = df[features].values
            y = df[target_variable].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Ensure at least 3 time steps for Conv1D by padding sequences
            X_train = pad_sequences(X_train, padding='post', maxlen=3, dtype='float32')
            X_test = pad_sequences(X_test, padding='post', maxlen=3, dtype='float32')

            # Reshape data for CNN & LSTM models
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Step 3: Select Model
            model_choice = st.radio("Select Model:", ["CNN", "LSTM", "Hybrid CNN-LSTM"])
            
            # Instantiate model
            if model_choice == "CNN":
                model = build_cnn_model(X_train.shape[1:])
            elif model_choice == "LSTM":
                model = build_lstm_model(X_train.shape[1:])
            else:
                model = build_hybrid_model(X_train.shape[1:])

            # Compile model
            model.compile(optimizer='adam', loss='mae')

            # Train model
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

            # Step 4: Evaluate Model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.write(f"📉 **Model Evaluation:** MAE = {mae:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

            # Step 5: Compute Water Quality Index (WQI)
            wqi = (1 / (1 + np.exp(-y_pred))).mean() * 100  # Example calculation
            st.write(f"💧 **Estimated Water Quality Index (WQI):** {wqi:.2f}")

            # Step 6: Provide Insights
            st.write("**🔍 Insights**")

            # Insights based on WQI
            if wqi > 80:
                st.success("✅ **Water quality is excellent!** Minimal pollutants detected.")
                st.write("The water is safe for aquatic life and human activities. Continue regular monitoring to maintain quality.")
            elif wqi > 50:
                st.warning("⚠️ **Water quality is moderate.** Some pollutants may be present.")
                st.write("Consider implementing measures such as aeration, filtration, or reducing nutrient runoff to improve water quality.")
            else:
                st.error("❌ **Poor water quality!** High pollutant levels detected.")
                st.write("Immediate action is required to address pollution sources. Consider advanced treatment methods and stricter regulations.")

        with col2:
            st.subheader(f"📊 Interactive Scatter Plot: {target_variable} Predictions vs Actual")

            # Create interactive scatter plot
            fig_scatter = px.scatter(
                x=y_test, y=y_pred.flatten(),
                labels={"x": "Actual Values", "y": f"Predicted {target_variable}"},
                title=f"Scatter Plot for {target_variable} (R² = {r2:.4f})"
            )

            # Add vertical reference line for mean actual values
            fig_scatter.add_vline(x=np.mean(y_test), line_dash="dash", line_color="red", annotation_text="Mean of Actual Values")

            # Overlay line for actual values
            fig_scatter.add_trace(
                px.line(
                    x=y_test, y=y_test,  # Actual values plotted as a line
                    labels={"x": "Actual Values", "y": target_variable}
                ).data[0]
            )

            # Update legend and layout
            fig_scatter.update_layout(
                legend_title=f"Prediction vs Actual {target_variable}",
                legend=dict(x=0.8, y=1, font=dict(size=12, color="black"))
            )

            # Display plot
            st.plotly_chart(fig_scatter, use_container_width=True)

    fig_line = px.line(
        x=df["Date"].iloc[-len(y_test):],  # Align with test set
        y=[y_test, y_pred.flatten()],
        labels={"x": "Date", "y": target_variable},
        title=f"Time-Series Plot for {target_variable}"
    )

    # Manually update trace colors (Actual = Red, Predicted = Blue)
    fig_line.data[0].line.color = "red"   # Actual values
    fig_line.data[1].line.color = "blue"  # Predicted values

    # Update layout for better visualization
    fig_line.update_layout(
        legend_title=f"Trend Comparison: {target_variable}",
        legend=dict(x=0.8, y=1, font=dict(size=12, color="black"))
    )

    # Display the plot
    st.plotly_chart(fig_line, use_container_width=True)


# --- Page 4: Recommendations ---
def page4():
    st.title("Recommendations")
    st.write("We have taken notes fo the insights given to us and based on the analysis, here are some recommendations to improve water quality in Taal Lake:")

    st.markdown("""
    - **Regular Monitoring:** Implement continuous monitoring of water quality parameters
    - **Data-Driven Interventions:** Use improved predictive models to forecast water quality and improve the models' performance.
    - **Daily Updates:** Provide daily monuitoring to track changes in water quality accurately.
    - **Data Collection:** Collect data from PHILVOCS and PAGASA to properly predict volcanic activity and weather conditions that may affect water quality.
    - **Community Engagement:** Involve local communities in monitoring and conservation efforts.
    - **Pollution Control:** Enforce stricter regulations on waste disposal and other contaminants discharged into the lake.
    - **Community Awareness:** Educate local communities about the importance of water conservation and pollution prevention.
    - **Restoration Projects:** Initiate projects to restore natural habitats around the lake to improve biodiversity.
    - **Sustainable Practices:** Promote sustainable agricultural practices to reduce runoff pollutants.
    """)

members = {
    "Augustine Sengodayan": {
        "role": "📊 Lead Researcher",
        "expertise": "Data analysis & water quality modeling",
        "photo": r"C:\Users\August\Documents\CPEN 106\MC02\Augustine Sengodayan.JPG",  # Use raw string
        "bio": "Born on October 13, 2004, Augustine is a passionate third-year Computer Engineering student at Cavite State University. Augustine resides in Trece Martires City, Philippines, where they are actively working on innovative projects related to data mining and water quality prediction."
    },
}

def page5():
    st.title("About")
    
    for name, details in members.items():
        col1, col2 = st.columns([2, 3])  # Adjust column width ratio as needed

        # Column 1: Photo
        with col1:
            st.image(details["photo"], caption=name, use_container_width=True)  # Corrected "True"

        # Column 2: Details
        with col2:
            st.subheader(name)
            st.write(f"**Role:** {details['role']}")
            st.write(f"**Expertise:** {details['expertise']}")
            st.write(f"**Bio:** {details['bio']}")


pages = [
    st.Page(page1, icon="🏠", title="Home"),
    st.Page(page2, icon="📊", title="Exploratory Data Analysis"),
    st.Page(page3, icon="🔬", title="Predictions"),
    st.Page(page4, icon="📖", title="Recommendations"),
    st.Page(page5, icon="💁", title="About"),
]
current_page = st.navigation(pages=pages, position="hidden")

# Set page layout
st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size: 100px !important; /* Increased font size */
    font-family: Helvetica, sans-serif !important;
    font-weight: bold !important;
    text-align: center !important;
    margin-bottom: 0px !important; /* Remove space below the heading */
}
.center-text {
    text-align: center !important;
    font-size: 20px !important;
    margin-top: 0px !important; /* Remove space above the subheading */
    margin-bottom: 30px !important; /* Add space below the heading */
}
hr {
    margin-top: 0px !important; /* Remove space above the line */
    margin-bottom: 0px !important; /* Remove space below the line */
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">TaalIntel</p>', unsafe_allow_html=True)
st.markdown('<p class="center-text">Data Mining and Data Visualization for Water Quality Prediction in Taal Lake</p>', unsafe_allow_html=True)
# Create a navigation bar
with st.container():
    columns = st.columns(len(pages) + 1)
    columns[0].write("**Water Quality Prediction Dashboard**")

    for i, page in enumerate(pages):
        columns[i + 1].page_link(page, icon=page.icon)

# Add a horizontal line with no space
st.markdown('<hr>', unsafe_allow_html=True)

current_page.run()