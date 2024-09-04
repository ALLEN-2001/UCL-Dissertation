#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
import pandas as pd

# Data based on the provided table
data = {
    "Time Interval": ["8:30-10:30", "9:00-10:30", "9:30-10:30", "10:00-10:30"],
    "Random Forest": [2.87, 3.0, 2.79, 3.45],
    "Cnn-LSTM": [2.90, 2.97, 2.87, 2.95],
    "LSTM": [3.08, 2.99, 3.10, 3.07],
    "SVM": [3.51, 2.98, 3.90, 4.28]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Setting the plot
plt.figure(figsize=(10, 5))

# Plotting each model's performance across time intervals
plt.plot(df["Time Interval"], df["Random Forest"], marker='o', label='Random Forest', color='#e79fe4')
plt.plot(df["Time Interval"], df["Cnn-LSTM"], marker='o', label='Cnn-LSTM', color='#9770bf')  # HEX code for green
plt.plot(df["Time Interval"], df["LSTM"], marker='o', label='LSTM', color='#8a87d0')  # RGB for blue
plt.plot(df["Time Interval"], df["SVM"], marker='o', label='SVM', color='#90b5a4')  # Gray color

plt.title('Root mean square errors with different forecast times.')
plt.xlabel('Time Interval')
plt.ylabel('Performance Metric')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd

# 更新后的数据
data = {
    "Time Interval": ["8:30-10:30", "9:00-10:30", "9:30-10:30", "10:00-10:30"],
    "Random Forest": [1.93, 1.98, 1.98, 2.5],
    "Cnn-LSTM": [2.20, 2.16, 2.29, 2.24],
    "LSTM": [2.62, 2.79, 2.77, 2.79],
    "SVM": [2.87, 2.51, 3.51, 3.55]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置绘图
plt.figure(figsize=(10, 5))

# 绘制每个模型在不同时间间隔的表现
plt.plot(df["Time Interval"], df["Random Forest"], marker='o', label='Random Forest', color='#e79fe4')
plt.plot(df["Time Interval"], df["Cnn-LSTM"], marker='o', label='Cnn-LSTM', color='#9770bf')
plt.plot(df["Time Interval"], df["LSTM"], marker='o', label='LSTM', color='#8a87d0')
plt.plot(df["Time Interval"], df["SVM"], marker='o', label='SVM', color='#90b5a4')

plt.title('Mean Absolute Errors with Different Forecast Times')
plt.xlabel('Time Interval')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()


# In[4]:


import matplotlib.pyplot as plt
import pandas as pd

# Data based on the provided table
data = {
    "Time Interval": ["08:00- 10:30", "08:30- 10:30", "09:00- 10:30", "09:30- 10:30"],
    "Random Forest": [3.31, 3.046, 2.75, 3.41],
    "Cnn-LSTM": [2.40, 2.97, 2.87, 2.95],
    "LSTM": [3.08, 2.99, 3.10, 3.07],
    "SVM": [3.51, 2.98, 3.90, 4.28]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Setting the plot
plt.figure(figsize=(10, 5))

# Plotting each model's performance across time intervals
plt.plot(df["Time Interval"], df["Random Forest"], marker='o', label='Random Forest', color='#e79fe4')
plt.plot(df["Time Interval"], df["Cnn-LSTM"], marker='o', label='Cnn-LSTM', color='#9770bf')  # HEX code for green
plt.plot(df["Time Interval"], df["LSTM"], marker='o', label='LSTM', color='#8a87d0')  # RGB for blue
plt.plot(df["Time Interval"], df["SVM"], marker='o', label='SVM', color='#90b5a4')  # Gray color

plt.title('Root mean square errors with different forecast times.')
plt.xlabel('Time Interval')
plt.ylabel('Performance Metric')
plt.legend()
plt.grid(True)
plt.show()


# In[1]:


import matplotlib.pyplot as plt
import pandas as pd

# Data based on the provided table
data = {
    "Metrics": ["RMSE", "MAE", "R2"],
    "Random Forest": [3.51, 3.17, -0.04],
    "Cnn-LSTM": [3.34, 2.87, 0.086],
    "LSTM": [3.49, 3.06, 0.13],
    "SVM": [3.97, 2.79, -0.34]
}

# Creating DataFrame
df = pd.DataFrame(data).set_index("Metrics")

# Setting up the plot
plt.figure(figsize=(10, 6))

# Plotting each model's performance metrics with specific colors
plt.plot(df.index, df["Random Forest"], marker='o', label='Random Forest', color='#e79fe4')
plt.plot(df.index, df["Cnn-LSTM"], marker='o', label='Cnn-LSTM', color='#9770bf')
plt.plot(df.index, df["LSTM"], marker='o', label='LSTM', color='#8a87d0')
plt.plot(df.index, df["SVM"], marker='o', label='SVM', color='#90b5a4')

# Adding titles and labels
plt.title('Comparison of Models Across Various Metrics')
plt.xlabel('Metric')
plt.ylabel('Metric Value')
plt.legend(title='Model')
plt.grid(True)
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data
rmse = [3.47, 3.34, 3.49, 3.76]
mae = [3.13, 2.87, 3.06, 3.19]
r2 = [-0.02, 0.086, 0.13, -0.199]

# Adjusting the spacing between the bars by increasing the bar_width
bar_width = 0.2  # Reduced the width to increase spacing

# Create the plot with the requested configuration
fig, ax = plt.subplots(figsize=(10, 6))

# Define the positions of the bars based on the metrics (RMSE, MAE, R^2)
x = np.arange(3)

# Plotting each model's metrics with their own color
ax.bar(x - bar_width*1.5, [rmse[0], mae[0], r2[0]], width=bar_width, label='Random Forest',color='#e79fe4')
ax.bar(x - bar_width/2, [rmse[1], mae[1], r2[1]], width=bar_width, label='CNN-LSTM',color='#9770bf')
ax.bar(x + bar_width/2, [rmse[2], mae[2], r2[2]], width=bar_width, label='LSTM',color='#8a87d0')
ax.bar(x + bar_width*1.5, [rmse[3], mae[3], r2[3]], width=bar_width, label='SVM',color='#90b5a4')

# Adding titles and labels
ax.set_title('Performance of the four models under the three indicators')
# ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(['RMSE', 'MAE', 'R^2'])

# Adding a legend
ax.legend()

plt.tight_layout()
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# Define the data
rmse = [3.47, 3.34, 3.49, 3.76]
mae = [3.13, 2.87, 3.06, 3.19]
r2 = [-0.02, 0.086, 0.13, -0.199]

# Adjusting the spacing between the bars by increasing the bar_width
bar_width = 0.2  # Reduced the width to increase spacing

# Create the plot with the requested configuration
fig, ax = plt.subplots(figsize=(10, 6))

# Define the positions of the bars based on the metrics (RMSE, MAE, R^2)
x = np.arange(3)

# Plotting each model's metrics with their own color
bars1 = ax.bar(x - bar_width*1.5, [rmse[0], mae[0], r2[0]], width=bar_width, label='Random Forest', color='#e79fe4')
bars2 = ax.bar(x - bar_width/2, [rmse[1], mae[1], r2[1]], width=bar_width, label='CNN-LSTM', color='#9770bf')
bars3 = ax.bar(x + bar_width/2, [rmse[2], mae[2], r2[2]], width=bar_width, label='LSTM', color='#8a87d0')
bars4 = ax.bar(x + bar_width*1.5, [rmse[3], mae[3], r2[3]], width=bar_width, label='SVM', color='#90b5a4')

# Adding titles and labels
ax.set_title('Performance of the four models under the three indicators')
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(['RMSE', 'MAE', 'R^2'])

# Adding a legend
ax.legend()

# Adding the data values on top of the bars
def add_values(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

# Apply the add_values function to each group of bars
add_values(bars1)
add_values(bars2)
add_values(bars3)
add_values(bars4)

plt.tight_layout()
plt.show()


# In[5]:


import matplotlib.pyplot as plt

# Data from the table provided by the user
time_intervals = ['8:00-10:30', '8:30-10:30', '9:00-10:30', '9:30-10:30']
random_forest = [3.31, 3.046, 2.75, 3.41]
cnn_lstm = [2.4, 2.97, 2.87, 2.95]
lstm = [3.08, 2.99, 3.10, 3.07]
svm = [3.51, 4.3, 3.90, 4.53]

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(time_intervals, random_forest, marker='s', label='Random Forest')
plt.plot(time_intervals, cnn_lstm, marker='o', label='CNN-LSTM')
plt.plot(time_intervals, lstm, marker='^', label='LSTM')
plt.plot(time_intervals, svm, marker='v', label='SVM')

# Adding titles and labels
plt.title('Root mean square errors with different forecast times.')
plt.ylabel('RMSE')

# Adding a legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()


# In[6]:


import matplotlib.pyplot as plt

# Data from the table provided by the user
time_intervals = ['8:00-10:30', '8:30-10:30', '9:00-10:30', '9:30-10:30']
random_forest = [4.31, 4.78, 5.2, 5.28]
cnn_lstm = [3.45, 4.51 , 4.56, 4.61]
lstm = [4.67, 5.3, 5.5, 5.59]
svm = [5.1, 5.71, 6.0, 6.1]

random_forest = [5.28, 5.2, 4.78, 4.31]
cnn_lstm = [4.61, 4.56 , 4.51, 3.45]
lstm = [5.59, 5.5, 5.3, 4.67]
svm = [6.1, 6.1, 5.71, 5.1]


# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(time_intervals, random_forest, marker='s', label='Random Forest')
plt.plot(time_intervals, cnn_lstm, marker='o', label='CNN-LSTM')
plt.plot(time_intervals, lstm, marker='^', label='LSTM')
plt.plot(time_intervals, svm, marker='v', label='SVM')

# Adding titles and labels
plt.title('Mean absolute errors with different forecast times.')
plt.ylabel('MAE')

# Adding a legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据，包含实际值和不同模型的预测值
data = {
    'Actual': [96, 95, 93, 85, 93, 95, 96, 96, 96, 81, 90, 95, 96, 93, 96, 92, 91, 87, 94, 96, 95, 93, 95, 85, 91, 92, 94, 95, 95, 95, 89, 91, 91, 94],
    'Random_Forest': [92.53, 89.79, 86.93, 86.75, 92.34, 92.88, 92.57, 92.89, 92.87, 88.84, 88.12, 92.65, 92.79, 89.35, 92.65, 89.57, 88.64, 88.41, 92.07, 92.74, 92.53, 89.09, 92.08, 90.59, 85.98, 87.56, 88.44, 92.51, 92.76, 91.62, 89.28, 86.52, 88.24, 92.22],
    'CNN_LSTM': [93.59225, 92.431816, 91.042885, 89.61689, 89.53068, 90.56542, 94.57473, 94.32867, 94.25207, 93.94636, 92.6793, 91.72862, 90.276855, 91.03035, 90.86534, 93.396194, 91.85708, 89.61901, 89.67813, 89.85757, 93.39475, 92.49253, 93.292915, 93.17927, 91.45624, 89.188866, 91.73424, 89.230934, 91.67019, 93.47129, 90.581696, 90.63342, 90.88457, 90.63286],
    'SVM': [88.36964402, 90.43114116, 94.66051523, 93.51828922, 93.56401641, 91.41698744, 93.43899725, 95.51475766, 98.60971225, 87.93150106, 87.23270485, 90.77447803, 90.10197972, 93.74593984, 94.83287588, 94.07312932, 89.27684617, 87.57396862, 89.41274907, 90.1270718, 90.61345294, 94.99333554, 92.00372568, 89.95722486, 89.98990732, 89.88038709, 90.60316507, 89.79837061, 92.17258294, 93.7462273, 93.19072835, 93.89199544, 91.79889728, 89.5234014],
    'LSTM': [94.485825, 92.10009, 90.198166, 91.01792, 91.11522, 91.619316, 92.0642, 92.86854, 91.685585, 91.604034, 90.3396, 91.75029, 92.63563, 93.82899, 91.59946, 92.1254, 88.97293, 89.15902, 89.770294, 91.89922, 93.62762, 93.85177, 92.33459, 90.64554, 91.34193, 92.004684, 92.584114, 93.35497, 93.29624, 92.361206, 91.05296, 91.69296, 92.55649, 92.228396]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 计算误差
df['Error_RF'] = df['Random_Forest'] - df['Actual']
df['Error_CNN_LSTM'] = df['CNN_LSTM'] - df['Actual']
df['Error_SVM'] = df['SVM'] - df['Actual']
df['Error_LSTM'] = df['LSTM'] - df['Actual']

# 绘制误差分布图
plt.figure(figsize=(14, 10))

# 随机森林误差分布
plt.subplot(2, 2, 1)
sns.histplot(df['Error_RF'], kde=True, color='#e79fe4')
plt.title('Random Forest Error Distribution')
plt.xlabel('Error')

# CNN-LSTM误差分布
plt.subplot(2, 2, 2)
sns.histplot(df['Error_CNN_LSTM'], kde=True, color='#9770bf')
plt.title('CNN-LSTM Error Distribution')
plt.xlabel('Error')

# SVM误差分布
plt.subplot(2, 2, 3)
sns.histplot(df['Error_SVM'], kde=True, color='#8a87d0')
plt.title('SVM Error Distribution')
plt.xlabel('Error')

# LSTM误差分布
plt.subplot(2, 2, 4)
sns.histplot(df['Error_LSTM'], kde=True, color='#90b5a4')
plt.title('LSTM Error Distribution')
plt.xlabel('Error')

plt.tight_layout()
plt.show()

# 绘制残差图
plt.figure(figsize=(14, 10))

# 随机森林残差图
plt.subplot(2, 2, 1)
sns.residplot(x=df['Actual'], y=df['Random_Forest'], lowess=True, color='#e79fe4')
plt.title('Random Forest Residual Plot')
plt.xlabel('Actual')
plt.ylabel('Residuals')

# CNN-LSTM残差图
plt.subplot(2, 2, 2)
sns.residplot(x=df['Actual'], y=df['CNN_LSTM'], lowess=True, color='#9770bf')
plt.title('CNN-LSTM Residual Plot')
plt.xlabel('Actual')
plt.ylabel('Residuals')

# SVM残差图
plt.subplot(2, 2, 3)
sns.residplot(x=df['Actual'], y=df['SVM'], lowess=True, color='#8a87d0')
plt.title('SVM Residual Plot')
plt.xlabel('Actual')
plt.ylabel('Residuals')

# LSTM残差图
plt.subplot(2, 2, 4)
sns.residplot(x=df['Actual'], y=df['LSTM'], lowess=True, color='#90b5a4')
plt.title('LSTM Residual Plot')
plt.xlabel('Actual')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据，包含实际值和不同模型的预测值
data = {
    'Actual': [96, 95, 93, 85, 93, 95, 96, 96, 96, 81, 90, 95, 96, 93, 96, 92, 91, 87, 94, 96, 95, 93, 95, 85, 91, 92, 94, 95, 95, 95, 89, 91, 91, 94],
    'Random_Forest': [92.53, 89.79, 86.93, 86.75, 92.34, 92.88, 92.57, 92.89, 92.87, 88.84, 88.12, 92.65, 92.79, 89.35, 92.65, 89.57, 88.64, 88.41, 92.07, 92.74, 92.53, 89.09, 92.08, 90.59, 85.98, 87.56, 88.44, 92.51, 92.76, 91.62, 89.28, 86.52, 88.24, 92.22],
    'CNN_LSTM': [93.59225, 92.431816, 91.042885, 89.61689, 89.53068, 90.56542, 94.57473, 94.32867, 94.25207, 93.94636, 92.6793, 91.72862, 90.276855, 91.03035, 90.86534, 93.396194, 91.85708, 89.61901, 89.67813, 89.85757, 93.39475, 92.49253, 93.292915, 93.17927, 91.45624, 89.188866, 91.73424, 89.230934, 91.67019, 93.47129, 90.581696, 90.63342, 90.88457, 90.63286],
    'SVM': [88.36964402, 90.43114116, 94.66051523, 93.51828922, 93.56401641, 91.41698744, 93.43899725, 95.51475766, 98.60971225, 87.93150106, 87.23270485, 90.77447803, 90.10197972, 93.74593984, 94.83287588, 94.07312932, 89.27684617, 87.57396862, 89.41274907, 90.1270718, 90.61345294, 94.99333554, 92.00372568, 89.95722486, 89.98990732, 89.88038709, 90.60316507, 89.79837061, 92.17258294, 93.7462273, 93.19072835, 93.89199544, 91.79889728, 89.5234014],
    'LSTM': [94.485825, 92.10009, 90.198166, 91.01792, 91.11522, 91.619316, 92.0642, 92.86854, 91.685585, 91.604034, 90.3396, 91.75029, 92.63563, 93.82899, 91.59946, 92.1254, 88.97293, 89.15902, 89.770294, 91.89922, 93.62762, 93.85177, 92.33459, 90.64554, 91.34193, 92.004684, 92.584114, 93.35497, 93.29624, 92.361206, 91.05296, 91.69296, 92.55649, 92.228396]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 计算误差
df['Error_RF'] = df['Random_Forest'] - df['Actual']
df['Error_CNN_LSTM'] = df['CNN_LSTM'] - df['Actual']
df['Error_SVM'] = df['SVM'] - df['Actual']
df['Error_LSTM'] = df['LSTM'] - df['Actual']

# 绘制误差分布图和残差图，按模型并排放置
models = ['Random_Forest', 'CNN_LSTM', 'SVM', 'LSTM']
errors = ['Error_RF', 'Error_CNN_LSTM', 'Error_SVM', 'Error_LSTM']
colors = ['#e79fe4', '#9770bf', '#8a87d0', '#90b5a4']

plt.figure(figsize=(16, 12))

for i, model in enumerate(models):
    # 绘制误差分布图
    plt.subplot(4, 2, 2*i+1)
    sns.histplot(df[errors[i]], kde=True, color=colors[i])
    plt.title(f'{model} Error Distribution')
    plt.xlabel('Error')
    
    # 绘制残差图
    plt.subplot(4, 2, 2*i+2)
    sns.residplot(x=df['Actual'], y=df[model], lowess=True, color=colors[i])
    plt.title(f'{model} Residual Plot')
    plt.xlabel('Actual')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()


# In[12]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2

# 假设数据已加载并预处理
# Set target column and other relevant settings
target_column = 'Timeliness (%)'
n_input_steps = 3
lstm_units = 100
dropout_rate = 0.3
l2_rate = 0.01
output_units = 1

# Function to create CNN-LSTM model
def create_regularized_model(input_shape, lstm_units, output_units, dropout_rate=0.2, l2_rate=0.01):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(lstm_units, activation='relu', kernel_regularizer=l2(l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Rolling Window Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)  # 5 splits for rolling cross-validation

# Store RMSEs for comparison
rmse_original_model = []
rmse_lagged_model = []

# Iterate over each split
for train_index, test_index in tscv.split(train_data_processed):
    train_split = train_data_processed.iloc[train_index]
    test_split = train_data_processed.iloc[test_index]

    # Preparing data
    X_train_split = train_split.drop(columns=[target_column])
    y_train_split = train_split[target_column]
    X_test_split = test_split.drop(columns=[target_column])
    y_test_split = test_split[target_column]

    # Normalize data (use scaler fitted on train data to transform both train and test)
    scaler = MinMaxScaler()
    X_train_split_scaled = scaler.fit_transform(X_train_split)
    X_test_split_scaled = scaler.transform(X_test_split)
    
    # Prepare sequence data for LSTM
    X_train_seq, y_train_seq = [], []
    for i in range(n_input_steps, len(X_train_split_scaled)):
        X_train_seq.append(X_train_split_scaled[i-n_input_steps:i, :])
        y_train_seq.append(y_train_split.iloc[i])
    
    X_test_seq, y_test_seq = [], []
    for i in range(n_input_steps, len(X_test_split_scaled)):
        X_test_seq.append(X_test_split_scaled[i-n_input_steps:i, :])
        y_test_seq.append(y_test_split.iloc[i])

    X_train_seq, y_train_seq = np.array(X_train_seq), np.array(y_train_seq)
    X_test_seq, y_test_seq = np.array(X_test_seq), np.array(y_test_seq)

    # Define and train the model
    model = create_regularized_model((X_train_seq.shape[1], X_train_seq.shape[2]), lstm_units, output_units, dropout_rate, l2_rate)
    model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=16, verbose=0)

    # Predict and calculate RMSE
    y_pred_seq = model.predict(X_test_seq)
    y_pred_seq_rescaled = scaler.inverse_transform(y_pred_seq.reshape(-1, 1))
    y_test_seq_rescaled = y_test_seq.reshape(-1, 1)

    rmse = np.sqrt(mean_squared_error(y_test_seq_rescaled, y_pred_seq_rescaled))
    rmse_original_model.append(rmse)

# Calculate the average RMSE for the original model across the rolling windows
avg_rmse_original_model = np.mean(rmse_original_model)
print(f'Average RMSE for Original Model: {avg_rmse_original_model}')

# Repeat similar steps for the lagged feature model
# Assuming you have generated lagged features as shown earlier
# We'll now implement the same process for the lagged model

for train_index, test_index in tscv.split(train_data_processed):
    train_split = train_data_processed.iloc[train_index]
    test_split = train_data_processed.iloc[test_index]

    # Preparing data with lagged features
    X_train_split_lagged = train_split.drop(columns=[target_column])
    y_train_split_lagged = train_split[target_column]
    X_test_split_lagged = test_split.drop(columns=[target_column])
    y_test_split_lagged = test_split[target_column]

    # Normalize data (use scaler fitted on train data to transform both train and test)
    X_train_split_lagged_scaled = scaler.fit_transform(X_train_split_lagged)
    X_test_split_lagged_scaled = scaler.transform(X_test_split_lagged)
    
    # Prepare sequence data for LSTM
    X_train_seq_lagged, y_train_seq_lagged = [], []
    for i in range(n_input_steps, len(X_train_split_lagged_scaled)):
        X_train_seq_lagged.append(X_train_split_lagged_scaled[i-n_input_steps:i, :])
        y_train_seq_lagged.append(y_train_split_lagged.iloc[i])
    
    X_test_seq_lagged, y_test_seq_lagged = [], []
    for i in range(n_input_steps, len(X_test_split_lagged_scaled)):
        X_test_seq_lagged.append(X_test_split_lagged_scaled[i-n_input_steps:i, :])
        y_test_seq_lagged.append(y_test_split_lagged.iloc[i])

    X_train_seq_lagged, y_train_seq_lagged = np.array(X_train_seq_lagged), np.array(y_train_seq_lagged)
    X_test_seq_lagged, y_test_seq_lagged = np.array(X_test_seq_lagged), np.array(y_test_seq_lagged)

    # Define and train the model
    model_lagged = create_regularized_model((X_train_seq_lagged.shape[1], X_train_seq_lagged.shape[2]), lstm_units, output_units, dropout_rate, l2_rate)
    model_lagged.fit(X_train_seq_lagged, y_train_seq_lagged, epochs=10, batch_size=16, verbose=0)

    # Predict and calculate RMSE
    y_pred_seq_lagged = model_lagged.predict(X_test_seq_lagged)
    y_pred_seq_lagged_rescaled = scaler.inverse_transform(y_pred_seq_lagged.reshape(-1, 1))
    y_test_seq_lagged_rescaled = y_test_seq_lagged.reshape(-1, 1)

    rmse_lagged = np.sqrt(mean_squared_error(y_test_seq_lagged_rescaled, y_pred_seq_lagged_rescaled))
    rmse_lagged_model.append(rmse_lagged)

# Calculate the average RMSE for the lagged model across the rolling windows
avg_rmse_lagged_model = np.mean(rmse_lagged_model)
print(f'Average RMSE for Lagged Feature Model: {avg_rmse_lagged_model}')

# Compare the results
if avg_rmse_lagged_model < avg_rmse_original_model:
    print("The model with lagged features performed better.")
else:
    print("The original model performed better.")


# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据

# 如果需要，转换时间戳（基于之前的上下文）
if 'time' in data.columns:
    data['time'] = pd.to_datetime(data['time'])

# 计算与 'Timeliness (%)' 相关的相关矩阵
correlation_matrix = data.corr()

# 绘制热图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate lagged features
for lag in range(1, 25):    
    data[f'lag_{lag}'] = data['Timeliness (%)'].shift(lag)

# Generate rolling averages
data['rolling_mean_3'] = data['Timeliness (%)'].rolling(window=3).mean()
# data['rolling_mean_7'] = data['Timeliness (%)'].rolling(window=7).mean()
data['rolling_mean_14'] = data['Timeliness (%)'].rolling(window=14).mean()

# Drop rows with NaN values created by lagging/rolling
data.dropna(inplace=True)

# Filter data for training and testing
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-05-14')]
test_data = data[(data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30')]

# Set target column
target_column = 'Timeliness (%)'

# Preprocess training data
n_input_steps = 3  # Using past 5 days data
train_data_processed = train_data.drop(columns=['time'])  # Remove time column
train_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize training data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data_processed)

# Target variable scaler
scaler_target = MinMaxScaler()
scaler_target.fit(train_data_processed[[target_column]])

# Prepare training data sequences
target_index = train_data_processed.columns.get_loc(target_column)
X_train, y_train = [], []
for i in range(n_input_steps, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-n_input_steps:i, :])
    y_train.append(scaled_train_data[i, target_index])
X_train, y_train = np.array(X_train), np.array(y_train)

# Preprocess test data
test_data_processed = test_data.drop(columns=['time'])  # Remove time column
test_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize test data
scaled_test_data = scaler.transform(test_data_processed)

# Prepare test data sequences
X_test, y_test = [], []
for i in range(n_input_steps, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-n_input_steps:i, :])
    y_test.append(scaled_test_data[i, target_index])
X_test, y_test = np.array(X_test), np.array(y_test)

# Define regularized CNN-LSTM model
def create_regularized_model(input_shape, lstm_units, output_units, dropout_rate=0.2, l2_rate=0.01):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(lstm_units, activation='relu', kernel_regularizer=l2(l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Set parameters with regularization
lstm_units = 100
dropout_rate = 0.3
l2_rate = 0.01
output_units = 1
input_shape = (X_train.shape[1], X_train.shape[2])

# Create and train regularized model
model = create_regularized_model(input_shape, lstm_units, output_units, dropout_rate, l2_rate)
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE

rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)



# Get test dates
test_dates = test_data['time'].iloc[n_input_steps:].reset_index(drop=True)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_rescaled, label='Actual Timeliness (%)', color='blue')
plt.plot(test_dates, y_pred_rescaled, label='Predicted Timeliness (%)', color='red')
plt.xlabel('Time')
plt.ylabel('Timeliness (%)')
plt.title('Comparison of Actual and Predicted Timeliness (%) (2024) on Regularized CNN-LSTM Model')
plt.legend()
plt.show()
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# 计算残差
residuals = y_test_rescaled.flatten() - y_pred_rescaled.flatten()

# 绘制残差图
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_rescaled, residuals, color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Timeliness (%)')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.grid(True)
plt.show()

# 检查残差的分布情况
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)
plt.show()


# In[ ]:


import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("Scikit-learn version:", sklearn.__version__)


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data

# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

#Filter data to keep only records from 8:00 to 10:30
data = data.set_index('time').between_time('08:30', '10:30').reset_index()

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate lagged features
for lag in range(1, 2):
    data[f'lag_{lag}'] = data['Timeliness (%)'].shift(lag)

# Generate rolling averages
data['rolling_mean_3'] = data['Timeliness (%)'].rolling(window=3).mean()
data['rolling_mean_7'] = data['Timeliness (%)'].rolling(window=7).mean()
data['rolling_mean_14'] = data['Timeliness (%)'].rolling(window=14).mean()

# Drop rows with NaN values created by lagging/rolling
data.dropna(inplace=True)

# Filter data for training and testing
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-04-30')]
test_data = data[(data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30')]


# Aggregate data to get only 10:30 records for prediction
test_data_10_30 = test_data[test_data['time'].dt.time == pd.to_datetime('09:45:00').time()]

# Set target column
target_column = 'Timeliness (%)'

# Preprocess training data
n_input_steps = 5
train_data_processed = train_data.drop(columns=['time'])  # Remove time column
train_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize training data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data_processed)

# Target variable scaler
scaler_target = MinMaxScaler()
scaler_target.fit(train_data_processed[[target_column]])

# Prepare training data sequences
target_index = train_data_processed.columns.get_loc(target_column)
X_train, y_train = [], []
for i in range(n_input_steps, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-n_input_steps:i, :])
    y_train.append(scaled_train_data[i, target_index])
X_train, y_train = np.array(X_train), np.array(y_train)

# Preprocess test data
test_data_processed = test_data.drop(columns=['time'])  # Remove time column
test_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize test data
scaled_test_data = scaler.transform(test_data_processed)

# Prepare test data sequences
X_test, y_test = [], []
for i in range(n_input_steps, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-n_input_steps:i, :])
    y_test.append(scaled_test_data[i, target_index])
X_test, y_test = np.array(X_test), np.array(y_test)

# Define regularized CNN-LSTM model
def create_regularized_model(input_shape, lstm_units, output_units, dropout_rate=0.2, l2_rate=0.01):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(lstm_units, activation='relu', kernel_regularizer=l2(l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Set parameters with regularization
lstm_units = 50
dropout_rate = 0.3
l2_rate = 0.01
output_units = 1
input_shape = (X_train.shape[1], X_train.shape[2])

# Create and train regularized model
model = create_regularized_model(input_shape, lstm_units, output_units, dropout_rate, l2_rate)
model.fit(X_train, y_train, epochs=100, batch_size=16)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Filter test data to get only 10:30 records
test_mask = test_data['time'].dt.time == pd.to_datetime('09:45:00').time()
y_pred_10_30 = y_pred_rescaled[test_mask[n_input_steps:]]
y_test_10_30 = y_test_rescaled[test_mask[n_input_steps:]]

# Extract only 10:30 predictions
test_dates_10_30 = test_data_10_30['time'].reset_index(drop=True)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_10_30, y_pred_10_30))

# Calculate MAE
mae = mean_absolute_error(y_test_10_30, y_pred_10_30)

# Calculate R2 score
r2 = r2_score(y_test_10_30, y_pred_10_30)


# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(test_dates_10_30, y_test_10_30, label='Actual Timeliness (%)', color='blue')
plt.plot(test_dates_10_30, y_pred_10_30, label='Predicted Timeliness (%)', color='red')
plt.xlabel('Time')
plt.ylabel('Timeliness (%)')
plt.legend()
plt.show()

print(f'RMSE: {rmse}')

print(f'MAE: {mae}')
print(f'R2: {r2}')


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate lagged features
# for lag in range(1, 25):
#     data[f'lag_{lag}'] = data['Timeliness (%)'].shift(lag)

# Generate rolling averages
data['rolling_mean'] = data['Timeliness (%)'].rolling(window=3).mean()

# Drop rows with NaN values created by lagging/rolling
data.dropna(inplace=True)

# Filter data for training and testing
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-05-14')]
test_data = data[(data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30')]

# Set target column
target_column = 'Timeliness (%)'

# Preprocess training data
n_input_steps = 3
train_data_processed = train_data.drop(columns=['time'])  # Remove time column
train_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize training data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data_processed)

# Target variable scaler
scaler_target = MinMaxScaler()
scaler_target.fit(train_data_processed[[target_column]])

# Prepare training data sequences
target_index = train_data_processed.columns.get_loc(target_column)
X_train, y_train = [], []
for i in range(n_input_steps, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-n_input_steps:i, :])
    y_train.append(scaled_train_data[i, target_index])
X_train, y_train = np.array(X_train), np.array(y_train)

# Preprocess test data
test_data_processed = test_data.drop(columns=['time'])  # Remove time column
test_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize test data
scaled_test_data = scaler.transform(test_data_processed)

# Prepare test data sequences
X_test, y_test = [], []
for i in range(n_input_steps, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-n_input_steps:i, :])
    y_test.append(scaled_test_data[i, target_index])
X_test, y_test = np.array(X_test), np.array(y_test)

# Define LSTM model
def create_lstm_model(input_shape, lstm_units, output_units, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=lstm_units, activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Set parameters
lstm_units = 50
dropout_rate = 0.2
output_units = 1
input_shape = (X_train.shape[1], X_train.shape[2])

# Create and train LSTM model
model = create_lstm_model(input_shape, lstm_units, output_units, dropout_rate)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

# Calculate MAE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

# Calculate R2 score
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

# Get test dates
test_dates = test_data['time'].iloc


print(f'RMSE: {rmse}')

print(f'MAE: {mae}')
print(f'R2: {r2}')


# In[ ]:


import matplotlib.pyplot as plt

# 绘制训练损失和验证损失随迭代次数的变化图
plt.figure(figsize=(14, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Plot of number of iterations against model fitness on LSTM')
plt.legend()
plt.show()


# In[ ]:


correlation_matrix = train_data_processed.corr()
target_corr = correlation_matrix[target_column]
print(target_corr.sort_values(ascending=False))


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate lagged features
for lag in range(1, 25):
    data[f'lag_{lag}'] = data['Timeliness (%)'].shift(lag)

# Generate rolling averages
data['rolling_mean_3'] = data['Timeliness (%)'].rolling(window=3).mean()

# Drop rows with NaN values created by lagging/rolling
data.dropna(inplace=True)

# Filter data for training and testing
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-05-14')]
test_data = data[(data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30')]

# Set target column
target_column = 'Timeliness (%)'

# Preprocess training data
n_input_steps = 5
train_data_processed = train_data.drop(columns=['time'])  # Remove time column
train_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize training data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data_processed)

# Target variable scaler
scaler_target = MinMaxScaler()
scaler_target.fit(train_data_processed[[target_column]])

# Prepare training data sequences
target_index = train_data_processed.columns.get_loc(target_column)
X_train, y_train = [], []
for i in range(n_input_steps, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-n_input_steps:i, :])
    y_train.append(scaled_train_data[i, target_index])
X_train, y_train = np.array(X_train), np.array(y_train)

# Preprocess test data
test_data_processed = test_data.drop(columns=['time'])  # Remove time column
test_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize test data
scaled_test_data = scaler.transform(test_data_processed)

# Prepare test data sequences
X_test, y_test = [], []
for i in range(n_input_steps, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-n_input_steps:i, :])
    y_test.append(scaled_test_data[i, target_index])
X_test, y_test = np.array(X_test), np.array(y_test)

# Define LSTM model
def create_lstm_model(input_shape, lstm_units, output_units, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=lstm_units, activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Set parameters
lstm_units = 50
dropout_rate = 0.2
output_units = 1
input_shape = (X_train.shape[1], X_train.shape[2])

# Create and train LSTM model
model = create_lstm_model(input_shape, lstm_units, output_units, dropout_rate)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
# Calculate MAE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

# Calculate R2 score
r2 = r2_score(y_test_rescaled, y_pred_rescaled)


# Get test dates
test_dates = test_data['time'].iloc[n_input_steps:].reset_index(drop=True)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_rescaled, label='Actual Timeliness (%)', color='blue')
plt.plot(test_dates, y_pred_rescaled, label='Predicted Timeliness (%)', color='red')
plt.xlabel('Time')
plt.ylabel('Timeliness (%)')
plt.title('Comparison of Actual and Predicted Timeliness (%) (2024) on LSTM Model')
plt.legend()
plt.show()

# Plot training history
plt.figure(figsize=(14, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.show()


print(f'RMSE: {rmse}')

print(f'MAE: {mae}')
print(f'R2: {r2}')


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Filter data to keep only records from 9:00 to 10:30
data = data.set_index('time').between_time('09:00', '10:30').reset_index()

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate lagged features
for lag in range(1, 25):
    data[f'lag_{lag}'] = data['Timeliness (%)'].shift(lag)

# Generate rolling averages
data['rolling_mean_3'] = data['Timeliness (%)'].rolling(window=3).mean()

# Shift the target variable to align with 9:45 prediction
data['target_9_45'] = data['Timeliness (%)'].shift(-5)  # Assuming 1-minute intervals

# Drop rows with NaN values created by lagging/rolling and shifting
data.dropna(inplace=True)

# Filter data for training and testing
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-05-14')]
test_data = data[(data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30')]

# Set target column to the shifted target
target_column = 'target_9_45'

# Preprocess training data
n_input_steps = 3
train_data_processed = train_data.drop(columns=['time'])  # Remove time column
train_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize training data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data_processed)

# Target variable scaler
scaler_target = MinMaxScaler()
scaler_target.fit(train_data_processed[[target_column]])

# Prepare training data sequences
target_index = train_data_processed.columns.get_loc(target_column)
X_train, y_train = [], []
for i in range(n_input_steps, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-n_input_steps:i, :])
    y_train.append(scaled_train_data[i, target_index])
X_train, y_train = np.array(X_train), np.array(y_train)

# Preprocess test data
test_data_processed = test_data.drop(columns=['time'])  # Remove time column
test_data_processed.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize test data
scaled_test_data = scaler.transform(test_data_processed)

# Prepare test data sequences
X_test, y_test = [], []
for i in range(n_input_steps, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-n_input_steps:i, :])
    y_test.append(scaled_test_data[i, target_index])
X_test, y_test = np.array(X_test), np.array(y_test)

# Define LSTM model
def create_lstm_model(input_shape, lstm_units, output_units, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=lstm_units, activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Set parameters
lstm_units = 50
dropout_rate = 0.2
output_units = 1
input_shape = (X_train.shape[1], X_train.shape[2])

# Create and train LSTM model
model = create_lstm_model(input_shape, lstm_units, output_units, dropout_rate)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Filter test data to get only 9:45 records
test_mask = test_data['time'].dt.time == pd.to_datetime('09:45:00').time()

# Ensure the mask and predictions align correctly
y_pred_9_45 = y_pred_rescaled[test_mask[n_input_steps:]]
y_test_9_45 = y_test_rescaled[test_mask[n_input_steps:]]

# Get test dates for 9:45 predictions
test_dates_9_45 = test_data['time'][test_mask].reset_index(drop=True)

# Ensure matching dimensions
min_length = min(len(test_dates_9_45), len(y_test_9_45), len(y_pred_9_45))
test_dates_9_45 = test_dates_9_45[:min_length]
y_test_9_45 = y_test_9_45[:min_length]
y_pred_9_45 = y_pred_9_45[:min_length]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_9_45, y_pred_9_45))

# Calculate MAE
mae = mean_absolute_error(y_test_9_45, y_pred_9_45)

# Calculate R2 score
r2 = r2_score(y_test_9_45, y_pred_9_45)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(test_dates_9_45, y_test_9_45, label='Actual Timeliness (%)', color='blue')
plt.plot(test_dates_9_45, y_pred_9_45, label='Predicted Timeliness (%)', color='red')
plt.xlabel('Time')
plt.ylabel('Timeliness (%)')
plt.legend()
plt.show()

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 假设数据和模型已经训练好
# 此处的X_train_scaled, y_train, X_test_scaled, y_test为之前的预处理数据

# 初始化数组来记录每棵树的累计预测
cumulative_predictions = np.zeros(len(X_test_scaled))

# 初始化列表来存储每棵树的RMSE
rmse_per_tree = []

# 逐棵树进行预测并计算RMSE
for i, tree in enumerate(rf.estimators_):
    # 累计预测
    cumulative_predictions += tree.predict(X_test_scaled)
    
    # 计算当前的平均预测值
    current_average_prediction = cumulative_predictions / (i + 1)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test, current_average_prediction))
    
    # 将RMSE添加到列表中
    rmse_per_tree.append(rmse)

# 绘制RMSE随树数增加的变化图
plt.figure(figsize=(14, 6))
plt.plot(range(1, len(rf.estimators_) + 1), rmse_per_tree, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Trees in Random Forest')
plt.show()


# In[ ]:


# 获取特征的重要性
importances = rf.feature_importances_

# 创建特征重要性数据框
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 打印特征重要性
print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()  # 倒转y轴使得最重要的特征在顶部
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate rolling averages
data['rolling_mean_3'] = data['Timeliness (%)'].rolling(window=3).mean()
data['rolling_mean_7'] = data['Timeliness (%)'].rolling(window=7).mean()
data['rolling_mean_14'] = data['Timeliness (%)'].rolling(window=14).mean()

# Drop rows with NaN values created by lagging/rolling
data.dropna(inplace=True)

# Filter training data for 9:30 AM to 10:30 AM
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-04-30')]
train_data = train_data[(data['time'].dt.hour == 9) & (data['time'].dt.minute >= 30) | (data['time'].dt.hour == 10) & (data['time'].dt.minute <= 30)]

# Filter test data for exactly 10:30 AM
test_data = data[(data['time'].dt.hour == 9) & (data['time'].dt.minute == 45) & (data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30')]


# Set target column
target_column = 'Timeliness (%)'
features = [col for col in train_data.columns if col != target_column and col != 'time']

# Extract features and target
X_train = train_data[features]
y_train = train_data[target_column]
X_test = test_data[features]
y_test = test_data[target_column]

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf.predict(X_test_scaled)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)


# Get test dates
test_dates = test_data['time'].reset_index(drop=True)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test, label='Actual Timeliness (%)', color='blue')
plt.plot(test_dates, y_pred, label='Predicted Timeliness (%)', color='red')
plt.xlabel('Time')
plt.ylabel('Timeliness (%)')
plt.title('Comparison of Actual and Predicted Timeliness (%) (2024) on Random Forest Model')
plt.legend()
plt.show()

print(f'RMSE: {rmse}')

print(f'MAE: {mae}')
print(f'R2: {r2}')


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate lagged features
for lag in range(1, 10):
    data[f'lag_{lag}'] = data['Timeliness (%)'].shift(lag)

# Generate rolling averages
data['rolling_mean_4'] = data['Timeliness (%)'].rolling(window=10).mean()

# Drop rows with NaN values created by lagging/rolling
data.dropna(inplace=True)

# Filter data for training and testing
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-05-14')]
test_data = data[(data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30')]

# Set target column
target_column = 'Timeliness (%)'
features = [col for col in train_data.columns if col != target_column and col != 'time']

# Extract features and target
X_train = train_data[features]
y_train = train_data[target_column]
X_test = test_data[features]
y_test = test_data[target_column]

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalize the target
scaler_target = MinMaxScaler()
y_train_scaled = scaler_target.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1)).ravel()

# Train the SVM model
svm_model = SVR(kernel='rbf', C=50, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train_scaled)

# Make predictions
y_pred_scaled = svm_model.predict(X_test_scaled)

# Inverse transform predictions and actual values
y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_rescaled = scaler_target.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred))
# Calculate MAE
mae = mean_absolute_error(y_test_rescaled, y_pred)

# Calculate R2 score
r2 = r2_score(y_test_rescaled, y_pred)

# Get test dates
test_dates = test_data['time'].reset_index(drop=True)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_rescaled, label='Actual Timeliness (%)', color='blue')
plt.plot(test_dates, y_pred, label='Predicted Timeliness (%)', color='red')
plt.xlabel('Time')
plt.ylabel('Timeliness (%)')
plt.title('Comparison of Actual and Predicted Timeliness (%) (2024) on SVM Model')
plt.legend()
plt.show()

print(f'RMSE: {rmse}')

print(f'MAE: {mae}')
print(f'R2: {r2}')


# In[ ]:


# 准备不同的C值以进行实验
C_values = np.arange(1, 100, 10)  # 每隔10增加一个C值
rmse_values = []

for C in C_values:
    # 为每个C值创建一个新的SVM模型
    svm_model = SVR(kernel='rbf', C=C, gamma=0.1, epsilon=0.1)
    svm_model.fit(X_train_scaled, y_train_scaled)
    
    # 进行预测
    y_pred_scaled = svm_model.predict(X_test_scaled)
    y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred))
    rmse_values.append(rmse)

# 绘制C值与RMSE的关系图
plt.figure(figsize=(14, 6))
plt.plot(C_values, rmse_values, marker='o')
plt.xlabel('Hyperparameter variation')
plt.ylabel('RMSE')
plt.title('Plot of SVM performance against hyperparameter variation')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# 准备C值在1到500之间的变化
C_values = np.arange(1, 100, 10)  # 每隔10增加一个C值
rmse_values = []

for C in C_values:
    # 为每个C值创建一个新的SVM模型
    svm_model = SVR(kernel='rbf', C=C, gamma=0.1, epsilon=0.1)
    svm_model.fit(X_train_scaled, y_train_scaled)
    
    # 进行预测
    y_pred_scaled = svm_model.predict(X_test_scaled)
    y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred))
    rmse_values.append(rmse)

# 绘制C值与RMSE的关系图
plt.figure(figsize=(14, 6))
plt.plot(C_values, rmse_values, marker='o')
plt.xlabel('C value')
plt.ylabel('RMSE')
plt.title('RMSE vs C value in SVM Model')
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/allen/Desktop/Travel_Data_with_Rounded_Temperatures.csv'
data = pd.read_csv(file_path)

# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Add time features
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Generate lagged features
# for lag in range(1, 10):
#     data[f'lag_{lag}'] = data['Timeliness (%)'].shift(lag)

# Generate rolling averages
data['rolling_mean'] = data['Timeliness (%)'].rolling(window=20).mean()

# Drop rows with NaN values created by lagging/rolling
data.dropna(inplace=True)

# Set target column
target_column = 'Timeliness (%)'
features = [col for col in data.columns if col != target_column and col != 'time']

# Extract features and target
X = data[features]
y = data[target_column]

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train the SVM model
svm_model = SVR(kernel='rbf', C=50, gamma=0.1, epsilon=0.1)
svm_model.fit(X_scaled, y)

# Compute permutation importance
result = permutation_importance(svm_model, X_scaled, y, n_repeats=10, random_state=42)

# Get feature importances
importances = result.importances_mean

# Create a DataFrame for the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from SVM (Permutation Importance)')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


# In[ ]:


from sklearn.inspection import permutation_importance

# 计算置换重要性
result = permutation_importance(svm_model, X_test_scaled, y_test_scaled, n_repeats=10, random_state=42)

# 显示特征重要性
feature_importances = pd.Series(result.importances_mean, index=features)
feature_importances.sort_values(ascending=False, inplace=True)

print(feature_importances)

# 可视化
feature_importances.plot(kind='bar', figsize=(12, 6), color='skyblue')
plt.title('Feature Importance based on Permutation Importance')
plt.show()


# In[ ]:


#8:30 - 10:30
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/allen/Desktop/new 22-24.csv'
data = pd.read_csv(file_path)

# Convert timestamps
data['time'] = pd.to_datetime(data['time'])

# Add time features
data['hour'] = data['time'].dt.hour
data['minute'] = data['time'].dt.minute
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['weekday'] = data['time'].dt.weekday
data['is_weekend'] = data['time'].dt.weekday >= 5

# Filter training data for time between 8:30 and 10:30
train_data = data[(data['time'] >= '2022-03-23') & (data['time'] <= '2023-05-14') & 
                  (((data['hour'] == 8) & (data['minute'] >= 30)) | 
                   (data['hour'] == 9) | 
                   ((data['hour'] == 10) & (data['minute'] <= 30)))]

# Filter test data for only 10:30 each day
test_data = data[(data['time'] >= '2024-03-23') & (data['time'] <= '2024-04-30') & 
                 (data['hour'] == 9) & (data['minute'] == 45)]

# Generate lagged features
for lag in range(1, 2):
    train_data[f'lag_{lag}'] = train_data['Timeliness (%)'].shift(lag)
    test_data[f'lag_{lag}'] = test_data['Timeliness (%)'].shift(lag)

# Generate rolling averages
train_data['rolling_mean_3'] = train_data['Timeliness (%)'].rolling(window=3).mean()
test_data['rolling_mean_3'] = test_data['Timeliness (%)'].rolling(window=3).mean()

# Drop rows with NaN values created by lagging/rolling
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Set target column
target_column = 'Timeliness (%)'
features = [col for col in train_data.columns if col != target_column and col != 'time']

# Extract features and target
X_train = train_data[features]
y_train = train_data[target_column]
X_test = test_data[features]
y_test = test_data[target_column]

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalize the target
scaler_target = MinMaxScaler()
y_train_scaled = scaler_target.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1)).ravel()

# Train the SVM model
svm_model = SVR(kernel='rbf', C=50, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train_scaled)

# Make predictions
y_pred_scaled = svm_model.predict(X_test_scaled)

# Inverse transform predictions and actual values
y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_rescaled = scaler_target.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred))
# Calculate MAE
mae = mean_absolute_error(y_test_rescaled, y_pred)

# Calculate R2 score
r2 = r2_score(y_test_rescaled, y_pred)

# Get test dates
test_dates = test_data['time'].reset_index(drop=True)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_rescaled, label='Actual Timeliness (%)', color='blue')
plt.plot(test_dates, y_pred, label='Predicted Timeliness (%)', color='red')
plt.xlabel('Time')
plt.ylabel('Timeliness (%)')
plt.title('Comparison of Actual and Predicted Timeliness (%) (2024) on SVM Model')
plt.legend()
plt.show()

results = {
    'RMSE': rmse,
    'MAE': mae,
    'R2': r2
}

results

