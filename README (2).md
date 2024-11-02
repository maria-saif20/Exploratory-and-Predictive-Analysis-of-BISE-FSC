# Institutional Performance Analysis in Board Exams 2023

This repository contains a project for analyzing institutional performance in the BISE FSc exams of 2023 using **Exploratory Data Analysis (EDA)** and **K-Means Clustering**. The aim is to gain insights into grade distribution, passing percentages, and to classify institutions based on their performance.

## Dataset

The dataset used in this project includes details on institutional performance, such as:

- `Institutional_Code`: Unique ID for each institution.
- `Appeared`: Number of students who appeared for the exams.
- `Passed`: Number of students who passed.
- `Pass%`: Percentage of students who passed from each institution.
- Grades (A+, A, B, C, D, E): Counts of students achieving each grade.

### Files

- **`Institutional Result of Board Exams.csv`**: The dataset file with institutional performance data.
- **`analysis.ipynb`**: Jupyter Notebook file where EDA and clustering analyses are conducted.

---

## Analysis Overview

### 1. Exploratory Data Analysis (EDA)

This part of the project examines the data to understand the distribution of grades and passing percentages, as well as handling missing values.

#### Steps:
1. **Loading and Displaying Data**: Load the dataset and display basic information.
2. **Summary Statistics**: Generate summary statistics for an overview of each column.
3. **Grade Distribution Visualization**: Plot a bar chart showing the number of students achieving each grade.
4. **Passing Percentage Distribution**: Plot a histogram to visualize passing percentages across institutions.
5. **Handling Missing Values**: Identify and fill missing values with column means to ensure data consistency.

### 2. Clustering Analysis

This part of the project applies K-Means Clustering to classify institutions based on student appearance counts and passing percentages, helping identify similar performance patterns.

#### Steps:
1. **Data Scaling**: Scale data using StandardScaler for consistent units in clustering.
2. **K-Means Clustering**: Apply K-Means with 3 clusters to group institutions.
3. **Cluster Visualization**: Plot clusters on a scatter plot with passing percentage and student appearance counts.

---

## Installation and Setup

To run this project, you need Python and the following libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Usage

1. **Clone this repository**:
    ```bash
    git clone https://github.com/your-username/institutional-performance-analysis.git
    ```
2. **Load the Dataset**: Place `Institutional Result of Board Exams.csv` in the project directory.
3. **Run the Analysis**: Open `analysis.ipynb` in Jupyter Notebook and execute each cell to perform EDA and clustering analysis.
4. **Interpret Results**:
   - Review the summary statistics, visualizations, and clustering output for insights on institutional performance.

## Code Highlights

### EDA Code

#### Loading Data
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Institutional Result of Board Exams.csv')
print(data.head())
```

#### Grade Distribution Visualization
```python
grades = ['Grade A+', 'A', 'B', 'C', 'D', 'E']
grade_counts = data[grades].sum()

plt.figure(figsize=(10, 6))
sns.barplot(x=grade_counts.index, y=grade_counts.values, palette='viridis')
plt.title('Total Students Achieving Each Grade')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.show()
```

#### Missing Values Handling
```python
print(data.isnull().sum())
data.fillna(data.mean(), inplace=True)
print(data.isnull().sum())
```

### Clustering Code

#### Scaling Data and K-Means
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[['Pass%', 'Appeared']])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_
```

#### Cluster Visualization
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pass%', y='Appeared', hue='Cluster', data=data, palette='viridis')
plt.title('K-Means Clustering of Institutions')
plt.xlabel('Passing Percentage')
plt.ylabel('Number of Students Appeared')
plt.legend(title='Cluster')
plt.show()
```

---

## Results

- **Grade Distribution**: Visualizes the total number of students achieving each grade.
- **Passing Percentage Distribution**: Shows passing rates among institutions.
- **Cluster Analysis**: Groups institutions with similar performance for comparative insights.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License.
