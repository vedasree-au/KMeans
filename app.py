import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Performance Clustering",
    page_icon="📚",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("📚 Student Performance Clustering")
st.markdown("""
This project uses **KMeans Clustering** to classify students into:

- 🏆 Toppers
- 🙂 Average Students
- ⚠️ Risk Students
""")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("StudentsPerformance.csv")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------- FEATURE SELECTION ----------------

X = df[['math score', 'reading score', 'writing score']]

# ---------------- REMOVE OUTLIERS ----------------

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

IQR = Q3 - Q1

X = X[~((X < (Q1 - 1.0 * IQR)) |
        (X > (Q3 + 1.0 * IQR))).any(axis=1)]

# ---------------- SCALING ----------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------- ELBOW METHOD ----------------

st.subheader("📈 Elbow Method")

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots(figsize=(8,5))

ax1.plot(range(1,11), wcss, marker='o')

ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")

st.pyplot(fig1)

# ---------------- KMEANS ----------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

X['Cluster'] = clusters

# ---------------- LABEL CLUSTERS ----------------

cluster_means = X.groupby('Cluster').mean()

sorted_clusters = cluster_means.mean(axis=1).sort_values()

risk_cluster = sorted_clusters.index[0]
average_cluster = sorted_clusters.index[1]
topper_cluster = sorted_clusters.index[2]

cluster_names = {
    topper_cluster: "Topper",
    average_cluster: "Average Student",
    risk_cluster: "Risk Student"
}

X['Student Type'] = X['Cluster'].map(cluster_names)

# ---------------- CLUSTER VISUALIZATION ----------------

st.subheader("🎯 Student Clusters")

fig2, ax2 = plt.subplots(figsize=(10,7))

sns.scatterplot(
    x=X['math score'],
    y=X['reading score'],
    hue=X['Student Type'],
    palette='deep',
    s=120,
    ax=ax2
)

ax2.set_title("Student Performance Clustering")

st.pyplot(fig2)

# ---------------- FINAL OUTPUT ----------------

st.subheader("📋 Clustered Students")

st.dataframe(X)

# ---------------- CLUSTER SUMMARY ----------------

st.subheader("📊 Cluster Summary")

summary = X.groupby('Student Type')[[
    'math score',
    'reading score',
    'writing score'
]].mean()

st.dataframe(summary)