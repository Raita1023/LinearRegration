import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import messagebox, ttk

# Step 1: Load the dataset
data = pd.read_csv('smoking.csv')

# Step 2: Preprocess the data
X = data.select_dtypes(include=[np.number]).drop(columns=['smoking'])
y = data['smoking']
X = X.fillna(X.mean())  
y = y.fillna(y.mode()[0])  
X = (X - X.mean()) / X.std()  

# Step 3: Implementing Multivariable Linear Regression without SK-Learn
X_bias = np.c_[np.ones(X.shape[0]), X]  
theta = np.zeros(X_bias.shape[1])  

alpha = 0.01  
iterations = 1000  

# Define cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Define gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        theta -= (alpha / m) * X.T.dot(predictions - y)
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# Train the model using linear without SK-Learn implementation
theta, cost_history = gradient_descent(X_bias, y, theta, alpha, iterations)


# Step 4: Implementing Multivariable Linear Regression with SK-Learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Function to find closest match based on exact data match
def find_exact_user_match(user_input, dataset):
    for index, row in dataset.iterrows():
        row_features = row.drop(['smoking', 'ID'])
        if all(user_val == row_val for user_val, row_val in zip(user_input, row_features)):
            return row  
    return None

# Function to get smoking status for the user
def get_smoking_status(user_input):
    # Match the user input to the dataset
    matched_user = find_exact_user_match(user_input, data)

    if matched_user is not None:
        smoking_status = matched_user['smoking']
        return "Yes, this user is a smoker." if smoking_status == 1 else "No, this user is not a smoker."
    else:
        return "Sorry, the user is not found. Please try again."
    
    

# GUI Function to handle form submission
def on_submit():
    try:
        # Get user input for comparison range
        start_index = int(start_index_entry.get())
        num_users = int(num_users_entry.get())

        # Validate indices
        if start_index < 0 or start_index + num_users > len(data):
            messagebox.showerror("Input Error", "Invalid indices. Please enter a valid range.")
            return

        # Get results from both models
        sklearn_predictions = model.predict(X_test)
        without_sklearn_predictions = X_bias.dot(theta)

        # Step 5: Plot comparison for the selected users (Line Graph)
        plt.figure(figsize=(10, 6))

        # Plot the actual values
        plt.plot(range(start_index, start_index + num_users), y[start_index:start_index + num_users], label="Actual", color='black', alpha=0.5, marker='o')

        # Plot the predictions from the models
        plt.plot(range(start_index, start_index + num_users), without_sklearn_predictions[start_index:start_index + num_users], label="Without SK-Learn Predictions", color='red', alpha=0.5, marker='x')
        plt.plot(range(start_index, start_index + num_users), sklearn_predictions[start_index:start_index + num_users], label="SK-Learn Predictions", color='blue', alpha=0.5, marker='^')

        plt.xlabel("Sample")
        plt.ylabel("Smoking Status (0 = Non-Smoker, 1 = Smoker)")
        plt.legend()
        plt.title("Comparison of Predicted vs Actual Smoking Status")
        plt.grid(True)
        plt.show()

        # Calculate Mean Squared Error for both models
        mse_without_sklearn = mean_squared_error(y[start_index:start_index + num_users], without_sklearn_predictions[start_index:start_index + num_users])
        mse_sklearn = mean_squared_error(y[start_index:start_index + num_users], sklearn_predictions[start_index:start_index + num_users])

        result_text = f"Mean Squared Error (Without SK-Learn Model): {mse_without_sklearn}\n"
        result_text += f"Mean Squared Error (SK-Learn Model): {mse_sklearn}"

        result_label.config(text=result_text)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for start index and number of users.")

# GUI Function to check smoker status
def on_check_smoking_status():
    # Collect user input for smoker status check
    user_input = []

    # For each column (excluding 'smoking' and 'ID')
    for column in data.columns:
        if column not in ['smoking', 'ID']:  # Exclude 'smoking' and 'ID' columns
            if column in ['gender', 'oral', 'tartar']:  # Handle categorical columns
                valid_values = data[column].unique()
                value = user_input_entries[column].get()
                if value not in valid_values:
                    messagebox.showerror("Input Error", f"Invalid value for {column}. Please try again.")
                    return
            else:  # For numerical columns, take float input
                value = float(user_input_entries[column].get())
            user_input.append(value)

    # Get the smoking status for the entered data
    result = get_smoking_status(user_input)
    smoking_status_label.config(text=result)

# Step 6: Create the main window for the GUI
root = tk.Tk()
root.title("Smoking Status Prediction - Model Comparison")

# Step 7: Make the window resizable and adjust layout for full screen usage
root.geometry("1000x700")  # Window size will expand to 100% of screen width and height
root.configure(bg="white")

# Step 8: Create a canvas for scrolling
canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

# Configure canvas and scrollbar
canvas.configure(yscrollcommand=scrollbar.set)

# Create a window inside the canvas
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Add scrollbar to canvas
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# Step 9: Create a frame for input fields inside the scrollable frame
input_frame = tk.Frame(scrollable_frame, bg="white")
input_frame.pack(padx=10, pady=10, anchor="center")

# Step 10: Create a frame for displaying the result and graph
result_frame = tk.Frame(root, bg="white")
result_frame.pack(padx=20, pady=20, side="right", anchor="ne", fill="both", expand=True)

# Create a label and entry for start index
tk.Label(input_frame, text="Enter start index (user):", bg="white").pack(padx=10, pady=5)
start_index_entry = tk.Entry(input_frame)
start_index_entry.pack(padx=10, pady=5)

# Create a label and entry for number of users to compare
tk.Label(input_frame, text="Enter number of users to compare:", bg="white").pack(padx=10, pady=5)
num_users_entry = tk.Entry(input_frame)
num_users_entry.pack(padx=10, pady=5)

# Add submit button to trigger the comparison
submit_button = tk.Button(input_frame, text="Compare Models", command=on_submit)
submit_button.pack(pady=20)

# Label to display results of comparison
result_label = tk.Label(result_frame, text="", bg="white", font=("Helvetica", 12, "bold"))
result_label.pack(pady=20)

# Step 11: Add section for checking if a user is a smoker or not
tk.Label(input_frame, text="Enter user details to check smoker status:", bg="white").pack(padx=10, pady=10)

# Entry fields for user details
user_input_entries = {}
for column in data.columns:
    if column not in ['smoking', 'ID']:
        tk.Label(input_frame, text=column, bg="white").pack(pady=5)
        entry = tk.Entry(input_frame)
        entry.pack(pady=5)
        user_input_entries[column] = entry

# Button to check smoking status
check_button = tk.Button(input_frame, text="Check Smoking Status", command=on_check_smoking_status)
check_button.pack(pady=20)

# Label to display the smoking status result
smoking_status_label = tk.Label(result_frame, text="", bg="white", font=("Helvetica", 14, "bold"))
smoking_status_label.pack(pady=5)

# Step 12: Run the GUI
root.mainloop()
