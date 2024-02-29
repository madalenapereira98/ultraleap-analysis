import pandas as pd
import numpy as np
from scipy.stats import kstest
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def get_repo_path_in_notebook():
    """
    Finds path of repo from Notebook.
    Start running this once to correctly find
    other modules/functions
    """
    path = os.getcwd()
    repo_name = 'ultraleap_analysis'

    while path[-len(repo_name):] != 'ultraleap_analysis':

        path = os.path.dirname(path)

    return path

repo_path = get_repo_path_in_notebook()
code_path = os.path.join(repo_path, 'code')
os.chdir(code_path)


def kolm_smir_test(X_df, task, test, feat_file, to_save=False):
    """
    
    """
    if feat_file == 'new_features':
        ls_feat = ['num_events', 'mean_max_dist', 'sd_max_dist', 'coef_var_max_dist',
                   'slope_max_dist', 'decr_max_dist', 'mean_max_vel', 'sd_max_vel',
                    'coef_var_max_vel', 'slope_max_vel', 'mean_mean_vel', 'sd_mean_vel',
                    'coef_var_mean_vel', 'slope_mean_vel', 'mean_tap_dur', 'sd_tap_dur',
                    'coef_var_tap_dur', 'slope_tap_dur', 'mean_rms', 'sd_rms', 'slope_rms',
                    'sum_rms', 'jerkiness', 'entropy']
        
    elif feat_file == 'features':
        ls_feat = ['num_events', 'mean_max_dist', 'sd_max_dist', 'coef_var_max_dist', 
                    'slope_max_dist', 'decr_max_dist', 'max_open_vel', 'mean_open_vel', 
                    'sd_open_vel', 'coef_var_open_vel', 'slope_open_vel', 'max_close_vel', 
                    'mean_close_vel', 'sd_close_vel', 'coef_var_close_vel', 'slope_close_vel',
                    'mean_tap_dur', 'sd_tap_dur', 'coef_var_tap_dur', 'slope_tap_dur',
                    'mean_rms', 'sd_rms', 'slope_rms', 'sum_rms', 'jerkiness', 'entropy']

    # Check if all the features in ls_feat are present in X_df columns
    if all(feature in X_df.columns for feature in ls_feat):
        # Filter the DataFrame to keep only the desired columns in ls_feat
        X_df = X_df[ls_feat]
    else:
        # Handle the case when some features are missing in X_df
        print("Some features are missing in X_df.")

    # Store test statistic and p-values
    statistics = []
    p_values = []

    for column in X_df.columns:
        # standardize the data (to have mean 0, std. dev. 1) before applying the test
        standardized_data = (X_df[column] - X_df[column].mean()) / X_df[column].std()
        stat, p_value = kstest(standardized_data, 'norm')
        statistics.append(stat)
        p_values.append(p_value)

    # Create a DataFrame for easy visualization
    normality_df = pd.DataFrame({
        'Feature': X_df.columns,
        'Statistic': statistics,
        'P-Value': p_values
    })

    normality_df = normality_df.sort_values(by='Statistic', ascending=False)

    if to_save:
        file_name = f'{task}_{test}_{feat_file}.xlsx'
        stats_path = os.path.join(repo_path, 'statistical_analysis', task, test)

        if not os.path.exists(stats_path):
            os.makedirs(stats_path)

        normality_df.to_excel(os.path.join(stats_path,file_name))

        print('DataFrame is written to Excel File successfully.')
   
    return display(normality_df)



def krusk_wal_test(X_df, y, task, test, feat_file, to_save=False):
    
    """
    
    """

    results = []
    test = 'Kruskal_Wallis'

    if 'score' not in X_df.columns:
        X_df.insert(loc=X_df.shape[1],column='score',value=y)

    features = X_df.iloc[:,1:-1].columns

    for feature in features:
        groups = [X_df[X_df['score']==val][feature].values for val in X_df['score'].unique()]
        H, p_val = stats.kruskal(*groups)

        results.append({
            'feature': feature,
            'H_statistic': H,
            'p_value': p_val
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Sort the features by the H statistic in descending order
    results_df = results_df.sort_values(by='H_statistic', ascending=False)

    # Get the top 5 features
    top_6_features = results_df['feature'][:6]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

    for i, feature in enumerate(top_6_features):
        sns.boxplot(x='score', y=feature, data=X_df, ax=axs[i//3, i%3], palette='summer')
        axs[i//3, i%3].set_title(f'Box plot of {feature} by score')
        # sns.boxplot(x='score', y=feature, data=X_df, ax=axs[i])
        # axs[i].set_title(f'Box plot of {feature} by score')

    plt.tight_layout()
    plt.show()

    if to_save:
        fig_path = os.path.join(repo_path, 'statistical_analysis', task, test)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, f'{task}_top_6_features_{feat_file}'),
                                dpi=300,facecolor='w', bbox_inches='tight')

        file_name = f'{task}_{test}_{feat_file}.xlsx'
        stats_path = os.path.join(repo_path, 'statistical_analysis', task, test)

        if not os.path.exists(stats_path):
            os.makedirs(stats_path)

        results_df.to_excel(os.path.join(stats_path,file_name))

        print('DataFrame is written to Excel File successfully.')
    
    # Display the results
    display(results_df)
   
    return results_df


# Correlation matrix

def plot_correlation_matrix(X_df, task, feat_file, to_save=False):

    rename_task = rename_task_for_plotting(task)

    if feat_file == 'new_features':
        ls_feat = ['num_events', 'mean_max_dist', 'sd_max_dist', 'coef_var_max_dist',
                   'slope_max_dist', 'decr_max_dist', 'mean_max_vel', 'sd_max_vel',
                    'coef_var_max_vel', 'slope_max_vel', 'mean_mean_vel', 'sd_mean_vel',
                    'coef_var_mean_vel', 'slope_mean_vel', 'mean_tap_dur', 'sd_tap_dur',
                    'coef_var_tap_dur', 'slope_tap_dur', 'mean_rms', 'sd_rms', 'slope_rms',
                    'sum_rms', 'jerkiness', 'entropy']
    elif feat_file == 'features':
        ls_feat = ['num_events', 'mean_max_dist', 'sd_max_dist', 'coef_var_max_dist', 
                   'slope_max_dist', 'decr_max_dist', 'max_open_vel', 'mean_open_vel', 
                    'sd_open_vel', 'coef_var_open_vel', 'slope_open_vel', 'max_close_vel', 
                    'mean_close_vel', 'sd_close_vel', 'coef_var_close_vel', 'slope_close_vel',
                    'mean_tap_dur', 'sd_tap_dur', 'coef_var_tap_dur', 'slope_tap_dur',
                    'mean_rms', 'sd_rms', 'slope_rms', 'sum_rms', 'jerkiness', 'entropy']

    # Check if all the features in ls_feat are present in X_df columns
    if all(feature in X_df.columns for feature in ls_feat):
        # Filter the DataFrame to keep only the desired columns in ls_feat
        X_df = X_df[ls_feat]
    else:
        # Handle the case when some features are missing in X_df
        print("Some features are missing in X_df.")


    # load data and create correlation matrix
    corr_matrix = X_df.corr(method="spearman")

    # define mask to hide the upper triangle of the matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # create correlation matrix figure
    corr_matrix_fig = plt.figure(figsize=(10,8))

    # set the center of the colorscheme to 0 and define the color palette
    sns.heatmap(corr_matrix, annot=False, mask=mask, center=0, cmap='coolwarm')

    plt.title(f'Correlation Matrix of {rename_task}')

    if to_save:
        corr_matrix_path = os.path.join(repo_path, 'statistical_analysis', task, 'corr_matrix')
        if not os.path.exists(corr_matrix_path):
            os.makedirs(corr_matrix_path)
        corr_matrix_fig.savefig(os.path.join(corr_matrix_path, f'correlation_matrix_{task}'),
                                dpi=300,facecolor='w', bbox_inches='tight')

    plt.close()

    return corr_matrix

def rename_task_for_plotting(task):

    if task == 'ft':task='Finger Tapping'
    elif task == 'oc':task='Opening and Closing'
    elif task == 'ps': task='Pronation Supination'

    return task


###############################################################################################

def anova_test(X_df):
    X_df['condition'] = X_df['file'].apply(lambda x: x.split('_')[2] if len(x.split('_')) > 2 else None)
    # df is your dataframe
    # features is your list of feature names

    # Calculate corrected p-value threshold
    alpha = 0.05
    num_tests = len(X_df.iloc[:,1:-1].columns)
    corrected_alpha = alpha / num_tests

    # Create an empty DataFrame to store F-values, p-values, and significance status
    anova_table = pd.DataFrame(columns=['Feature', 'F-value', 'p-value', 'Significance'])

    for feature in X_df.iloc[:,1:-1].columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(x=X_df['condition'], y=feature, data=X_df.iloc[:,1:-1])
        
        groups = [X_df[X_df['condition']==val][feature].values for val in X_df['condition'].unique()]
        f_val, p_val = stats.f_oneway(*groups)
        
        # Test if p-value is significant with the corrected threshold
        significance = p_val < corrected_alpha
        signif_str = "Significant" if significance else "Not Significant"

        # Append the results to the anova_table
        anova_table = anova_table.append({
            'Feature': feature,
            'F-value': f_val,
            'p-value': p_val,
            'Significance': signif_str
        }, ignore_index=True)
        
        # Add corrected p-value to the plot
        plt.title(f"p-value = {p_val:.3f} ({signif_str})")

        plt.show()


    # Print the ANOVA table
    print(anova_table)

    return anova_table