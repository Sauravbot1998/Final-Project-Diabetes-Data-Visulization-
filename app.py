import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Diabetes Health Analysis Dashboard")

# 2. Load and Preprocess Data (Using relative path to avoid FileNotFoundError)
@st.cache_data
def load_data():
    # Relative path works on your local computer and Streamlit Cloud
    df = pd.read_csv('diabetes.csv')
    
    # Preprocessing for better visualization
    df['diabetes_label'] = df['diabetes'].map({0: 'Healthy', 1: 'Diabetic'})
    
    # Creating Age Groups (Requirement: Binning/Categorization)
    age_bins = [0, 30, 45, 60, 100]
    age_labels = ['Young Adult (0-30)', 'Middle Aged (31-45)', 'Senior (46-60)', 'Elderly (60+)']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Missing 'diabetes.csv'. Please place the CSV file in the same folder as this script.")
    st.stop()

# 3. Sidebar - Interactive Widgets (Required by rubric)
st.sidebar.header("Dashboard Filters")
st.sidebar.info("Adjust the filters below to explore different segments of the data.")

# Widget 1: Gender Filter
gender_filter = st.sidebar.multiselect("Select Gender", options=df['gender'].unique(), default=df['gender'].unique())

# Widget 2: Physical Activity Filter
activity_filter = st.sidebar.multiselect("Physical Activity Level", options=df['physical_activity'].unique(), default=df['physical_activity'].unique())

# Widget 3: BMI Slider
bmi_range = st.sidebar.slider("Select BMI Range", float(df['bmi'].min()), float(df['bmi'].max()), (15.0, 45.0))

# Apply Filters
filtered_df = df[
    (df['gender'].isin(gender_filter)) & 
    (df['physical_activity'].isin(activity_filter)) &
    (df['bmi'].between(bmi_range[0], bmi_range[1]))
]

# 4. Main Header
st.title("🩺 Diabetes Risk Factors: Visual Storytelling")
st.markdown("""
This dashboard analyzes 10,000 health records to identify key indicators of Diabetes. 
**Objective:** To answer 10 analytical questions regarding demographics, lifestyle, and biometrics.
""")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(filtered_df))
col2.metric("Avg Glucose", f"{filtered_df['glucose'].mean():.1f}")
col3.metric("Avg BMI", f"{filtered_df['bmi'].mean():.1f}")
col4.metric("Diabetes Rate", f"{(filtered_df['diabetes'].mean()*100):.1f}%")

st.divider()

# 5. The "10 Analytical Questions" Visualizations
# Question 1 & 2
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. How does Age impact Diabetes status?")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=filtered_df, x='age', hue='diabetes_label', kde=True, ax=ax1, palette='magma')
    st.pyplot(fig1)

with col_right:
    st.subheader("2. Is Family History a strong indicator?")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=filtered_df, x='family_history', hue='diabetes_label', ax=ax2)
    ax2.set_xticklabels(['No History', 'Has History'])
    st.pyplot(fig2)

# Question 3 & 4
st.divider()
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("3. What is the correlation between BMI and Glucose?")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='bmi', y='glucose', hue='diabetes_label', alpha=0.5, ax=ax3)
    st.pyplot(fig3)

with col_right2:
    st.subheader("4. How does Physical Activity affect BMI?")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=filtered_df, x='physical_activity', y='bmi', palette='Set2', ax=ax4)
    st.pyplot(fig4)

# Question 5 & 6
st.divider()
col_left3, col_right3 = st.columns(2)

with col_left3:
    st.subheader("5. Does Diet Score impact Cholesterol levels?")
    fig5, ax5 = plt.subplots()
    sns.lineplot(data=filtered_df, x='diet_score', y='cholesterol', hue='diabetes_label', ax=ax5)
    st.pyplot(fig5)

with col_right3:
    st.subheader("6. Distribution of Diabetes across Age Groups")
    fig6, ax6 = plt.subplots()
    sns.barplot(data=filtered_df, x='age_group', y='diabetes', ax=ax6, palette='viridis')
    plt.xticks(rotation=45)
    st.pyplot(fig6)

# Question 7 & 8
st.divider()
col_left4, col_right4 = st.columns(2)

with col_left4:
    st.subheader("7. How does Stress Level affect Sleep Hours?")
    fig7, ax7 = plt.subplots()
    sns.regplot(data=filtered_df.sample(500), x='stress_level', y='sleep_hours', scatter_kws={'alpha':0.3}, ax=ax7)
    st.pyplot(fig7)

with col_right4:
    st.subheader("8. Does Insulin level vary by Gender?")
    fig8, ax8 = plt.subplots()
    sns.violinplot(data=filtered_df, x='gender', y='insulin', hue='diabetes_label', split=True, ax=ax8)
    st.pyplot(fig8)

# Question 9 & 10 (Correlation Heatmap)
st.divider()
st.subheader("9. & 10. Global Feature Correlations & Heatmap")
st.write("Visualizing how all numerical health indicators interact with one another.")
fig9, ax9 = plt.subplots(figsize=(12,8))
numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax9)
st.pyplot(fig9)

# 6. Data Table (Rubric requirement)
st.divider()
st.subheader("Raw Data Inspection")
st.dataframe(filtered_df.head(50))