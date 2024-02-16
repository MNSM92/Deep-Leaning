from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# tools.py
#   A collection of useful functions
#   Created by: MNSM
#   Created on: 2024-01-19
#   Last edited: 2024-01-19
#   Last edited by: MNSM

# P(text, color) - Prints text in the specified color
def P(text, color):
    color_codes = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }

    if color not in color_codes:
        print("Invalid color. Please choose one of: red, green, yellow, blue, purple, cyan, white.")
        return

    print(f"{color_codes[color]}{text}{color_codes['reset']}")

# print important information
def show_df_info(data):
    P(f"Shape: \n{data.shape}", 'purple')
    P(f"Columns: \n{data.columns}", 'green')
    P(f"info: \n{data.info()}", 'green')
    P(f"describe: \n{data.describe()}", 'yellow')
    P(f"Index: \n{data.index}", 'purple')
    P(f"Data types: \n{data.dtypes}", 'yellow')
    P(f"Missing values: \n{data.isna().sum()}", 'white')
    P(f"Unique values: \n{data.nunique()}", 'blue')
    P(f"Memory usage: \n{data.memory_usage(deep=True).sum()}", 'purple')
    P(f"Head: \n{data.head()}", 'red')

    # if there is no missing values
    if data.isna().sum().sum() == 0:
        P("There are no missing values", 'green')
    else:
        P("There are missing values", 'red')


# Show Numerical analysis and visualization of the information of a column
def show_nav(data, column_name):
    P(f"Head: \n{data[column_name].head()}", 'red')
    P(f"Shape: \n{data[column_name].shape}", 'purple')
    P(f"Info: \n{data[column_name].info()}", 'green')
    P(f"Describe: \n{data[column_name].describe()}", 'yellow')
    P(f"Missing values: \n{data[column_name].isna().sum()}", 'white')
    P(f"Unique values: \n{data[column_name].nunique()}", 'blue')
    P(f"Data types: \n{data[column_name].dtypes}", 'purple')
    P(f"Memory usage: \n{data[column_name].memory_usage(deep=True)}", 'green')
    # plot the column
    data[column_name].plot(kind='box', vert=False, figsize=(14, 6))
    plt.savefig(f"./plots/{column_name}box.png")
    ax = data[column_name].plot(kind='density', figsize=(14,6)) # kde
    ax.axvline(data[column_name].mean(), color='red')
    ax.axvline(data[column_name].median(), color='green')
    plt.savefig(f"./plots/{column_name}density.png")
    data[column_name].plot(kind='hist', figsize=(14, 6))
    plt.savefig(f"./plots/{column_name}hist.png")

# Show Categorical analysis and visualization of the information of a column
def show_cav(data, column_name):
    P(f"Head: \n{data[column_name].head()}", 'red')
    P(f"Shape: \n{data[column_name].shape}", 'purple')
    P(f"Info: \n{data[column_name].info()}", 'green')
    P(f"Describe: \n{data[column_name].describe()}", 'yellow')
    P(f"Missing values: \n{data[column_name].isna().sum()}", 'white')
    P(f"Unique values: \n{data[column_name].nunique()}", 'blue')
    P(f"Data types: \n{data[column_name].dtypes}", 'purple')
    P(f"Memory usage: \n{data[column_name].memory_usage(deep=True)}", 'green')
    # plot the column
    data[column_name].value_counts().plot(kind='bar', figsize=(14,6))
    plt.savefig(f"./plots/{column_name}bar.png")
    data[column_name].value_counts().plot(kind='pie', figsize=(6,6))
    plt.savefig(f"./plots/{column_name}pie.png")

# Relationship between the columns
def show_corr(data):
    # Exclude non-numeric columns from correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr = numeric_data.corr()
    num_cols = len(numeric_data.columns)

    # Create a figure
    fig, axes = plt.subplots(num_cols, num_cols, figsize=(num_cols**2, num_cols**2))

    # Plot correlation matrix
    cax = axes[0,0].matshow(corr, cmap='RdBu')
    fig.colorbar(cax, ax=axes[0])
    axes[0,0].set_xticks(range(len(corr.columns)))
    axes[0,0].set_yticks(range(len(corr.columns)))
    axes[0,0].set_xticklabels(corr.columns, rotation='vertical')
    axes[0,0].set_yticklabels(corr.columns)
    axes[0,0].set_title('Correlation Matrix')

    # Plot scatter plot of all pairs of variables and make a single png
    for i, col1 in enumerate(numeric_data.columns):
        for j, col2 in enumerate(numeric_data.columns):
            if i != j:
                sns.scatterplot(data=data, x=col1, y=col2, ax=axes[i, j])
                axes[i, j].set_title(f'Scatter Plot: {col1} vs {col2}')

    # Save the plots
    plt.savefig("./plots/correlation.png")
