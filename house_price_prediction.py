import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('house_data.csv')

# Features and target
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Show predictions
predicted_df = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})
print(predicted_df)

# Optional Visualization
plt.scatter(df['SquareFeet'], df['Price'], color='blue', label='Actual')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('House Price vs Square Footage')
plt.legend()
plt.grid(True)
plt.show()
