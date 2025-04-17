# Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load Dataset
df = pd.read_csv('/Users/adityachaubey/Downloads/Crime_Incidents_in_2024.csv')

# Display Dataset Info
print("Dataset Size:", df.shape)
print("First 5 Rows:\n", df.head())
print(df.columns)


# Objective 1: Most Frequent Crime Types
plt.figure(figsize=(10, 6))
top_crimes = df['OFFENSE'].value_counts().head(10)
sns.barplot(x=top_crimes.values, y=top_crimes.index, color='salmon')
plt.title("Top 10 Most Common Crimes")
plt.xlabel("Number of Incidents")
plt.ylabel("Crime Type")
plt.tight_layout()
plt.show()

# Objective 2: High Crime Locations
plt.figure(figsize=(10, 6))
top_locations = df['BID'].value_counts().head(10)
sns.barplot(x=top_locations.values, y=top_locations.index, color="skyblue")
plt.title("Top 10 Crime-Prone Locations")
plt.xlabel("Number of Incidents")
plt.ylabel("Location")
plt.tight_layout()
plt.show()

# Objective 3: Temporal Crime Trends
df['datetime'] = pd.to_datetime(df['REPORT_DAT'])
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day_name()
df['month'] = df['datetime'].dt.month_name()

# Crimes by Hour
plt.figure(figsize=(10, 5))
sns.countplot(x='hour', data=df, color='mediumseagreen')

plt.title("Crimes by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# Crimes by Day
plt.figure(figsize=(10, 5))
sns.countplot(x='day', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], color='orange')
plt.title("Crimes by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# Objective 4: Crime Severity Visualization
plt.figure(figsize=(8, 5))
sns.countplot(x='SHIFT', data=df,color='salmon')
plt.title("Crime Distribution by Shift")
plt.xlabel("Shift (Time of Day)")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()


# Objective 5: Predict Crime Category
# Encode categorical variables
label_enc = LabelEncoder()
df['location_enc'] = label_enc.fit_transform(df['BID'])
df['shift_enc'] = label_enc.fit_transform(df['SHIFT'])
df['crime_type_enc'] = label_enc.fit_transform(df['OFFENSE'])  



# Prediction & Evaluation
X = df[['hour', 'location_enc', 'shift_enc']]
y = df['crime_type_enc']

