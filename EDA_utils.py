import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Debugging.
# from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')



def univariate_analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,    
                              outliers=False, kde=False, color='#8d0801', figsize=(24, 12)):
    '''
    Generate plots for univariate analysis.

    This function generates histograms, horizontal bar plots 
    and boxplots based on the provided data and features. 

    Args:
        data (DataFrame): The DataFrame containing the data to be visualized.
        features (list): A list of feature names to visualize.
        histplot (bool, optional): Generate histograms. Default is True.
        barplot (bool, optional): Generate horizontal bar plots. Default is False.
        mean (bool, optional): Generate mean bar plots of specified feature instead of proportion bar plots. Default is None.
        text_y (float, optional): Y coordinate for text on bar plots. Default is 0.5.
        outliers (bool, optional): Generate boxplots for outliers visualization. Default is False.
        kde (bool, optional): Plot Kernel Density Estimate in histograms. Default is False.
        color (str, optional): The color of the plot. Default is '#8d0801'.
        figsize (tuple, optional): The figsize of the plot. Default is (24, 12).

    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        # Get num_features and num_rows and iterating over the sublot dimensions.
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0) 
        
        fig, axes = plt.subplots(num_rows, 3, figsize=figsize)  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if barplot:
                if mean:
                    data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    bars = ax.barh(y=data_grouped[feature], width=data_grouped[mean], color=color)
                    for index, value in enumerate(data_grouped[mean]):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=15)
                else:
                    data_grouped = data.groupby([feature])[[feature]].count().rename(columns={feature: 'count'}).reset_index()
                    data_grouped['pct'] = round(data_grouped['count'] / data_grouped['count'].sum() * 100, 2)
                    bars = ax.barh(y=data_grouped[feature], width=data_grouped['pct'], color=color)
                    for index, value in enumerate(data_grouped['pct']):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=15)
                
                ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=15)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.grid(False)
                ax.get_xaxis().set_visible(False)
                
            elif outliers:
                # Plot univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax, color=color)

            else:
                # Plot histplot.
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='percent')

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
        # Remove unused axes.
        if num_features < len(axes.flat):
            for j in range(num_features, len(axes.flat)):
                fig.delaxes(axes.flat[j])

        plt.tight_layout()
    
    except Exception as e:
        print(e)
        #raise CustomException(e, sys)