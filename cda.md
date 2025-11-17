## 1. Basic R Op
```
# Letters
letters       # a–z
LETTERS       # A–Z

# Arithmetic operators
a <- 10
b <- 3
a + b
a - b
a * b
a / b
a %% b   # modulus
a ^ b    # power

# Relational operators
a > b
a < b
a == b
a != b

# Logical operators
TRUE & FALSE
TRUE | FALSE
!TRUE

# Assignment
x <- 5
y = 10
15 -> z

# Strings
name <- "Aditi"
nchar(name)
paste("Hello", name)

# Read string
# readline() cannot run in non-interactive mode, but example:
# str <- readline("Enter name: ")

```
## 2. Data types and Operations
```
# VECTORS
num_vec <- c(1, 2, 3, 4)
char_vec <- c("A", "B", "C")
log_vec <- c(TRUE, FALSE, TRUE)

seq_vec <- seq(1, 10, by = 2)

# Accessing
num_vec[2]
num_vec[2:4]
num_vec[num_vec > 2]

# Modifying
num_vec[1] <- 100

# Deleting
num_vec <- num_vec[-1]

# Sorting
sort(num_vec)

# LISTS
lst <- list(name="Aditi", age=20, marks=c(90, 80, 85))

# MATRICES
mat <- matrix(1:9, nrow=3, ncol=3)
mat[1, 2]

# Arrays
arr <- array(1:12, dim = c(2,2,3))

# Factors
colors <- factor(c("red","blue","red","green"))
levels(colors)

# DATA FRAME
df <- data.frame(
  Name=c("Aditi","Rahul"),
  Age=c(20,21),
  Marks=c(89,92)
)

df$Age
df[1,3]

# Add row
df <- rbind(df, c("Neha",22,95))

```
## 3. Sampling & Simulation in R
```
# Random sampling (numbers)
sample(1:20, 10)

# Sampling with replacement
sample(1:6, 4, replace = TRUE)

# Character sampling
sample(letters, 5)
sample(LETTERS, 26)

# Non-probability sampling example
vec <- c(10, 20, 30, 40, 50)
sample(vec, 3)                 # without replacement
sample(vec, 3, replace = TRUE) # with replacement

```
## 4. Mean, Median, Mode in R
```
marks <- c(45, 67, 78, 78, 90)

# Mean
mean(marks)

# Median
median(marks)

# Mode (custom function)
mode_func <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

mode <- function (){return(names(sort(-table(marks)))[1])}

mode_func(marks)
mode(marks)

# From CSV
# data <- read.csv("data.csv")
# mean(data$column)
# median(data$column)
# mode_func(data$column)

```
## 5. Range, SD, Variance, Percentiles, IQR
```
marks <- c(50,60,70,80,90,100)

# Range
range(marks)

# Standard deviation
sd(marks)

# Variance
var(marks)

# Percentile
quantile(marks, 0.70)

# Multiple percentiles
quantile(marks, c(0.7, 0.5, 0.8))

# Percentile for DF
df <- data.frame(Age = c(18,19,20,21,22,23))
quantile(df$Age, c(0.55, 0.27))

# Interquartile range
IQR(marks)
```
## 6. Data Visualization in R
```
# BARPLOT
values <- c(10, 20, 15, 30)
barplot(values, main="Vertical Bar Plot")

barplot(values, horiz=TRUE, main="Horizontal Bar Plot")

# HISTOGRAM
hist(values, main="Histogram", xlab="Values")

# BOX PLOT
boxplot(values, main="Box Plot")

# MULTIPLE BOXPLOTS
boxplot(values, rnorm(4), main="Multiple Box Plots")

# SCATTER PLOT
x <- 1:10
y <- x + rnorm(10)
plot(x, y, main="Scatter Plot")

# HEATMAP
mat <- matrix(rnorm(25), 5, 5)
heatmap(mat)

# MAPS
# install.packages("maps")
library(maps)
map("world")

# 3D GRAPH (persp)
x <- seq(-10, 10, length=30)
y <- x
z <- outer(x, y, function(a,b) sin(sqrt(a^2 + b^2)))
persp(x, y, z)
```
## 6.5 Date and Time in R
```
# Current date and time
Sys.Date()
Sys.time()

# Using lubridate
library(lubridate)
today()
now()

# Extract parts
dates <- as.Date(c("2023-03-10","2024-01-01"))
year(dates)
month(dates)
day(dates)

# Manipulate
new_date <- as.Date("2023-03-10") + 10

# Update using update()
update(ymd("2023-03-10"), year=2025, month=5)
```

## 7. Linear Reg
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate synthetic linear data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plot
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label="Actual")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Prediction Line")
plt.xlabel("X (Feature)")
plt.ylabel("Y (Target)")
plt.title("Simple Linear Regression (Synthetic Data)")
plt.legend()
plt.show()

```
## 8. Logistic Reg
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example dataset: hours studied vs pass/fail (1=Pass, 0=Fail)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot
X_range = np.linspace(0, 12, 100).reshape(-1, 1)
y_range_prob = model.predict_proba(X_range)[:, 1]

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X_range, y_range_prob, color='red', label="Logistic Curve")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Pass Prediction")
plt.legend()
plt.show()

```
## 9. Multiple Reg
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Dataset
data = {
    "Area_sqft": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
    "Bedrooms": [2, 3, 3, 4, 4, 5, 5, 6],
    "Distance_km": [10, 8, 6, 5, 4, 3, 2, 1],
    "Price": [200, 250, 280, 310, 360, 400, 450, 480]
}

df = pd.DataFrame(data)

X = df[["Area_sqft", "Bedrooms", "Distance_km"]]
y = df["Price"]

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predictions
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)   # Predictions for all samples

# Step 5: New unseen data prediction
new_data = pd.DataFrame({
    "Area_sqft": [2200, 2800, 5000],
    "Bedrooms": [3, 4, 6],
    "Distance_km": [7, 3, 2]
})
new_predictions = model.predict(new_data)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Mean Squared Error (Test):", mean_squared_error(y_test, y_pred_test))
print("R² Score (Test):", r2_score(y_test, y_pred_test))

print("\n--- Predictions for New Houses ---")
for i, price in enumerate(new_predictions):
    print(f"House {i+1}: Predicted Price = {price:.2f}")

# Step 6: Visualization
plt.figure(figsize=(12,5))

# Actual vs Predicted (All Data)
plt.subplot(1,2,1)
plt.plot(range(len(y)), y.values, marker='o', label="Actual", color="blue")
plt.plot(range(len(y)), y_pred_all, marker='s', label="Predicted", color="red")
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.title("Actual vs Predicted Prices (All Data)")
plt.legend()

```
## 10. Linear Reg with gradient
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load dataset (Boston Housing from online source)
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

print("Dataset shape:", data.shape)
print(data.head())

# Features (X) and Target (y)
X = data.drop("medv", axis=1).values  # features
y = data["medv"].values               # target (house price)

# 2. Standardize features for faster convergence
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term (x0 = 1)
X = np.c_[np.ones(X.shape[0]), X]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Gradient Descent for Linear Regression
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  # initialize weights
    cost_history = []

    for _ in range(epochs):
        y_pred = X.dot(theta)
        error = y_pred - y
        cost = (1/(2*m)) * np.sum(error ** 2)  # MSE cost
        cost_history.append(cost)

        gradients = (1/m) * X.T.dot(error)
        theta -= lr * gradients

    return theta, cost_history


# Run gradient descent
theta, cost_history = gradient_descent(X_train, y_train, lr=0.05, epochs=2000)

print("Final weights:", theta)
print("Final cost:", cost_history[-1])

# 4. Predictions
y_pred = X_test.dot(theta)

# 5. Plot cost convergence
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Convergence")
plt.show()

# 6. Plot Predictions vs Actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Prediction vs Actual")
plt.show()

```
## 11. Students' t test power analysis
```
from statsmodels.stats.power import TTestIndPower
effect = 0.8
alpha = 0.05
power = 0.8

analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f'% result)

from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower
effect_sizes = array([0.2,0.5,0.8])
sample_sizes = array(range(5,100))

analysis = TTestIndPower()
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()
```
