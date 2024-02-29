
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

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


## Creates box plots with seaborn
def boxplot(feature, boxplot_label, X_df, y, task):

    if 'score' not in X_df.columns:
        X_df.insert(loc=X_df.shape[1],column='score', value=y)

    path = os.path.join(repo_path, 'modelling', 'new_features', task, 'boxplots', boxplot_label)
    if not os.path.exists(path):
        os.makedirs(path)

    if boxplot_label == 'grouped_scores':
        ls_scores = [(0, 1), (2, 3, 4)]
    elif boxplot_label == 'scores':
        ls_scores = np.unique(y)
    elif boxplot_label == 'condition':
        ls_scores = ['m1s0', 'm1s1', 'm0s0', 'm0s1', 'm1']
    elif boxplot_label == 'camera':
        ls_scores = ['vr','dt','st']
        # X_df['condition'] = X_df['file'].apply(lambda x: x.split('_')[2] if len(x.split('_')) > 2 else None)
    else:
        raise ValueError("Invalid boxplot_label")

    # Create a list to store the data for each score
    data_by_score = []

    for score in ls_scores:
        if boxplot_label == 'scores':
            data = X_df[X_df['score'] == score][feature] # Wrap score in a list
        elif boxplot_label == 'grouped_scores':
            # for i in range(len(score)):
            data = X_df[X_df['score'].isin(score)][feature]
        elif boxplot_label == 'condition':
            # Extract the condition from the 'file' column and create a new 'condition' column
            # X_df['condition'] = X_df['file'].apply(lambda x: x.split('_')[2] if len(x.split('_')) > 2 else None)
            data = X_df[X_df['condition'] == score][feature]
        elif boxplot_label == 'camera':
            data = X_df[X_df['camera'] == score][feature]

        df = pd.DataFrame({
            boxplot_label: [score] * len(data),
            feature: data
        })
        data_by_score.append(df)

    # concatenate all dataframes
    data_df = pd.concat(data_by_score)

    # create the boxplot with seaborn
    plt.figure(figsize=(10,8))
    sns.boxplot(x=boxplot_label, y=feature, data=data_df, palette='summer')
    plt.title(f'Boxplot: {boxplot_label}')
    plt.savefig(os.path.join(path, f'box_plot_{feature}_{boxplot_label}'), dpi = 300)
    plt.close()

    return


## Different function that creates box plots with matplotlib
# def boxplot(feature, boxplot_label, X_df, y, task):

#     if 'score' not in X_df.columns:
#             X_df.insert(loc=X_df.shape[1],column='score', value=y)

#     path = os.path.join(repo_path, 'modelling', task, 'boxplots', boxplot_label)
#     if not os.path.exists(path):
#         os.makedirs(path)

#     if boxplot_label == 'grouped_scores':
#         # define the groups to plot
#         ls_scores = [(0, 1), (2, 3, 4)]
#         # Create a list to store the data for each score
#         data_by_score = {score: [] for score in ls_scores}

#         # Extract the data for each score
#         for score in ls_scores:
#             for i in range(len(score)):
#                 data = X_df[X_df['score']==score[i]][feature]
#                 data_by_score[score].extend(data)

#     elif boxplot_label == 'scores':
#         # define the scores to plot
#         ls_scores = np.unique(y)
#         # Create a list to store the data for each score
#         data_by_score = {s: [] for s in ls_scores}

#         # Extract the data for each score
#         for score in ls_scores:
#             data = X_df[X_df['score'] == score][feature]
#             data_by_score[score].extend(data)

#     elif boxplot_label == 'condition':
#         ls_scores = ['m1s0', 'm1s1', 'm0s0', 'm0s1', 'm1']
#         data_by_score = {c: [] for c in ls_scores}

#         # extract the data for each condition
#         for row in np.arange(0,X_df.shape[0]):
#             file_name = X_df['file'].iloc[row]
#             try:
#                 # Get the condition for this file
#                 cond = file_name.split('_')[2]
                
#                 # Extract the data for each condition in this file
#                 for c in ls_scores:
#                     data = X_df[(X_df['file'] == file_name) & (X_df[feature]) & (cond == c)][feature]
#                     data_by_score[c].extend(data)
                    
#             except IndexError:
#                 print(f'Invalid file name: {file_name}')



#     # create the boxplot
#     fig, ax = plt.subplots()
#     ax.boxplot(data_by_score.values())

#     # Set the title and labels for the x and y axes
#     ax.set_title(f'Boxplot: {boxplot_label}')
#     ax.set_xlabel(boxplot_label)
#     ax.set_ylabel(feature)

#     # Set the tick labels for the x axis
#     ax.set_xticklabels(data_by_score.keys())

#     fig.savefig(os.path.join(path,
#                                 f'boxplot_{feature}_{boxplot_label}'),
#                                 dpi = 300, facecolor = 'w'
#                                 )

#     plt.close()

#     return