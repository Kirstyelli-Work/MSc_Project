from matplotlib import cm
from IPython.display import display
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import numpy as np
import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from astronomaly.dimensionality_reduction import pca

def feature_pre_processing(features, data_root_dir):
    print('Features: ', features.shape)

    my_pca = pca.PCA_Decomposer(force_rerun=True, n_components=29, threshold=0.95, output_dir=data_root_dir)
    pca_features = my_pca.run(features)
    pca_features.to_parquet(os.path.join(data_root_dir, 'pca_features.parquet'))
    print('PCA Features: ', pca_features.shape)

    return pca_features

def label_pre_processing(full_volunteer_labels, data_root_dir, img_dir):

    full_volunteer_labels = full_volunteer_labels.set_index('iauname')
    print('Full GZD-5 Volunteer Labels: ', full_volunteer_labels.shape)
    full_volunteer_labels = full_volunteer_labels[~full_volunteer_labels['wrong_size_warning']]
    print('wrong_size_warning filtered GZD-5 Volunteer Labels: ', full_volunteer_labels.shape)

    my_label_df, summ = create_gz_evaluation_set(full_volunteer_labels, 
                                             question_level=3, min_votes=34, max_votes=None, half_votes_min=True, min_prob=0, max_prob=1, min_sources=100)
    print('Refined GZD-5 Evaluation Set: ', my_label_df.shape)

    ellipticals = my_label_df[(my_label_df['smooth-or-featured'] == 'smooth-or-featured_smooth') & 
                              (my_label_df['merging'] == 'merging_none') & 
                              (my_label_df['how-rounded'] == 'how-rounded_round')].index
    spirals = my_label_df[(my_label_df['smooth-or-featured'] == 'smooth-or-featured_featured-or-disk') & 
                          (my_label_df['merging'] == 'merging_none') &
                          (my_label_df['disk-edge-on'] == 'disk-edge-on_no')].index
    edge_ons = my_label_df[(my_label_df['smooth-or-featured'] == 'smooth-or-featured_featured-or-disk') & 
                           (my_label_df['merging'] == 'merging_none') &
                           (my_label_df['disk-edge-on'] == 'disk-edge-on_yes')].index

    sorting_all_images(ellipticals, 'elliptical', img_dir, data_root_dir)
    sorting_all_images(spirals, 'spiral', img_dir, data_root_dir)
    sorting_all_images(edge_ons, 'edge_on', img_dir, data_root_dir)

    # Combining the indices of the three classes to create a single DataFrame of volunteer labels
    volunteer_labels_indices = ellipticals.union(spirals).union(edge_ons)
    # Filtering the full volunteer labels to only include the galaxies that belong to the three classes
    volunteer_labels = full_volunteer_labels.loc[volunteer_labels_indices]
    # Assigning class labels to the volunteer labels DataFrame
    volunteer_labels.loc[ellipticals, 'Volunteer_Label'] = 'R'   # Round ellipticals
    volunteer_labels.loc[spirals, 'Volunteer_Label'] = 'S'       # Spirals
    volunteer_labels.loc[edge_ons, 'Volunteer_Label'] = 'E'      # Edge-ons

    volunteer_labels.to_parquet(os.path.join(data_root_dir, 'volunteer_labels.parquet'))

    print('Labelled Galaxies: ', volunteer_labels.shape)
    print('Round Ellipticals: ',    (volunteer_labels['Volunteer_Label'] == 'R').sum())
    print('Spirals: ',              (volunteer_labels['Volunteer_Label'] == 'S').sum())
    print('Edge-ons: ',             (volunteer_labels['Volunteer_Label'] == 'E').sum())

    return volunteer_labels

def myMBKM(features, n_clusters=20, max_iter=10, n_init="auto", metric_sample=None):
    # Perform fit
    my_kmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init).fit(features)
    #print('K-Means Iterations: ',my_kmeans.n_iter_)
    
    # Predict Labels, Centers and Distances
    kmeans_labels = my_kmeans.predict(features)
    kmeans_centers = my_kmeans.cluster_centers_
    kmeans_dist = my_kmeans.transform(features)
    kmeans_inertia = my_kmeans.inertia_

    # Metrics
    sample_features = features.iloc[metric_sample]
    sample_labels = kmeans_labels[metric_sample]

    kmeans_silhouette = silhouette_score(sample_features, sample_labels)
    kmeans_david = davies_bouldin_score(sample_features, sample_labels)

    # Create Distance Column Names
    dist_cols = []
    for i in range(kmeans_dist.shape[1]):
        dist_cols.append(f'Distance_{i}')
    # Saving cluster distance information
    clusters = pd.DataFrame(index = features.index.copy())
    clusters['Cluster'] = kmeans_labels
    clusters['Center'] = [list(kmeans_centers[i]) for i in kmeans_labels]
    clusters[dist_cols] = kmeans_dist

    return kmeans_inertia, kmeans_silhouette, kmeans_david, clusters

def myBGMM(features, n_components=20, weight_concentration_prior=0.5, n_init=10, max_iter=1000, metric_sample=None):
    # Perform fit
    my_bgmm = BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=weight_concentration_prior, n_init=n_init, max_iter=max_iter).fit(features)
    print("Converged: ", my_bgmm.converged_)
    print("BGMM Iterations: ", my_bgmm.n_iter_)
    # Predict Labels, Centers and Probabilities
    bgmm_labels = my_bgmm.predict(features)
    bgmm_centers = my_bgmm.means_
    bgmm_prob = my_bgmm.predict_proba(features)
    bgmm_weights = my_bgmm.weights_
    bgmm_lower_bound = my_bgmm.lower_bound_

    # Metrics
    sample_features = features.iloc[metric_sample]
    sample_labels = bgmm_labels[metric_sample]

    bgmm_silhouette = silhouette_score(sample_features, sample_labels)
    bgmm_david = davies_bouldin_score(sample_features, sample_labels)

    # Create Probability Column Names
    prob_cols = []
    for i in range(bgmm_prob.shape[1]):
        prob_cols.append(f'Prob_{i}')
    # Saving cluster probability information
    clusters = pd.DataFrame(index = features.index.copy())
    clusters['Cluster'] = bgmm_labels
    clusters['Center'] = [list(bgmm_centers[i]) for i in bgmm_labels]
    clusters[prob_cols] = bgmm_prob

    return bgmm_weights, bgmm_lower_bound, bgmm_silhouette, bgmm_david, clusters

def create_gz_evaluation_set(
        input_df, question_level=1, min_votes=0, max_votes=None, half_votes_min=True, min_prob=0, max_prob=1, min_sources=100):
    """
    Create a filtered evaluation set from a DataFrame of classification responses from Galaxy Zoo.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing classification responses, where each row corresponds to an object
        and columns include vote counts, probabilities, or other relevant features.

    question_level : int, default=1
        The depth level of the question in a decision tree or hierarchy to evaluate. For example,
        `question_level=1` may correspond to the top-level question.
    
    min_votes : int or None, optional
        Minimum number of total votes for first question required for a sample to be included. If `None`, no
        threshold is applied based on vote count. Recommended minimum from Walmsley et al is 34

    max_votes : int or None, optional
        Maximum number of total votes for first question  required for a sample to be included. If `None`, no
        threshold is applied based on vote count. This can be used to define an "uncertain" sample.
        
    half_votes_min : bool, default=True
        If `True`, ensures that at least half of the possible votes were cast for the level of
        question under consideration. Recommended to always be set to true.

    min_prob : float or None, optional
        Minimum probability threshold for including a classification. If `None`, no filtering 
        based on probability is applied.

    max_prob : float or None, optional
        Maximum probability threshold for including a classification. If `None`, no filtering 
        based on probability is applied. This can be used to define an "uncertain" sample.

    min_sources : int, optional
        Minimum number of sources before a specific combination of questions can be considered a class.

    Returns
    -------
    evaluation_set : pandas.DataFrame
        A filtered DataFrame including only the samples that meet the specified criteria.

    classes_summary : pandas.DatFrame
        A summary of each target with the unique set of answers and number of sources in that class.
    """
    # questions = [
    #     'smooth-or-featured','disk-edge-on', 'how-rounded','edge-on-bulge', 'bar', 'has-spiral-arms', 'merging','bulge-size', 'spiral-winding', 'spiral-arm-count']

    levels_dict = { 
                1:['smooth-or-featured'],
                2:['merging'],
                3:['disk-edge-on', 'how-rounded'],
                4:['edge-on-bulge','has-spiral-arms', 'bar', 'bulge-size'], 
                5:['spiral-winding', 'spiral-arm-count']
              }
    ignore_list = ['total-votes', 'fraction', 'debiased']
    questions   = []

    for lvl in levels_dict.keys():
        if lvl <= question_level:
            questions += levels_dict[lvl]
    
    # Step 0: Filter rows based on total votes for the first question
    msk = input_df['smooth-or-featured_total-votes'] >= min_votes
    if max_votes is not None:
        msk = msk*(input_df['smooth-or-featured_total-votes'] < max_votes)
    df = input_df.loc[msk]

    answers_df = pd.DataFrame(index=df.index)

    for q in questions:
        # Step 1: Identify answer columns
        answer_cols = []
        for c in df.columns:
            # Check if the query string appears in the column name, if so, add it to answer_cols
            if q in c:
                suffix = c.split('_')[-1] # Get the last part of the column name
                if suffix not in ignore_list:
                    answer_cols.append(c)
        
        # Step 2: Get vote values and best answer per row
        vote_vals = df[answer_cols].values  # shape (n_rows, n_answers)
        best_idx = np.argmax(vote_vals, axis=1)  # shape (n_rows,)
        best_answers = np.array(answer_cols)[best_idx]  # shape (n_rows,)
        best_votes = vote_vals[np.arange(len(df)), best_idx] # shape (n_rows,)
        total_votes = vote_vals.sum(axis=1)
        
        # Step 3: Get corresponding fraction values using NumPy indexing        
        best_fractions = np.zeros(len(df))
        msk = total_votes != 0
        best_fractions[msk] = best_votes[msk]/total_votes[msk]
    
        # Step 4: Apply Probability Filter
        prob_cut_mask = (best_fractions < min_prob) | (best_fractions > max_prob)
        best_answers[prob_cut_mask] = 'N/A'

        # Step 5: Apply Half Votes Filter
        if half_votes_min is True:
            half_vote_cut = total_votes < 0.5 * df['smooth-or-featured_total-votes'] #Why do we trust this one though?
            best_answers[half_vote_cut] = 'N/A'

        # Step 6: Store best answers in the answers_df
        answers_df[q] = best_answers
        
    # Step 7: Only keep rows where each level has at least one valid answer (tag)
    for lvl in range(1, question_level+1): 
        answers_df = answers_df[~np.all(answers_df[levels_dict[lvl]]=='N/A', axis=1)]

    # Step 8: Assign target labels to each class and create classes summary dataframe
    counts = answers_df.value_counts() # Get counts of each unique combination of answers
    counts = counts[counts>min_sources]
    top_rows = counts.reset_index() # Convert to DataFrame and reset index
    top_rows['target'] = range(len(top_rows)) # Assign target labels as integers starting from 0
    classes_summary = pd.DataFrame(top_rows)
    classes_summary.set_index('target', inplace=True)
    
    # Step 9: Merge df_reset (modified answers_df) with top rows to assign target labels to each row in the original answers_df
    # (if the combination of answers is in the top rows, it gets the target label, otherwise it is dropped)
    df_reset = answers_df.reset_index() # Adding column of iauname from index of answers_df
    top_rows = top_rows.drop(columns=['count'])  # Drop the count column
    df_top = df_reset.merge(top_rows, on=list(answers_df.columns), how='inner')
    df_top = df_top.set_index('iauname') # Restore original index
    
    # Step 10: Adding complexity column showing no. of tags for each target class
    classes_summary['complexity'] = (classes_summary[classes_summary.columns[:-1]]!='N/A').sum(axis=1)
    
    return df_top, classes_summary

def save_random_images(folder, classification, data_root_dir):

    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    random_images = random.sample(files, 5)

    fig, axes = plt.subplots(1, 5, figsize=(10,2))

    for ax, img_name in zip(axes.flatten(), random_images):

        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path)
        label = os.path.splitext(img_name)[0]

        ax.imshow(img)
        ax.text(0.98, 0.98, label,
                transform=ax.transAxes,
                ha='right', va='top',
                color='white', fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(data_root_dir, f'Galaxy_Images/{classification}_random_images.png'))
    plt.close()
    print(f'Random images for {classification} saved in {os.path.join(data_root_dir, f"Galaxy_Images/{classification}_random_images.png")}')

def sorting_all_images(classification, class_name,img_dir, data_root_dir):
    
    filenames_to_find = set(classification.astype(str) + '.png')
    count = 0

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file in filenames_to_find:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(data_root_dir, f'Galaxy_Images/{class_name}', file)
                shutil.copy2(src_path, dest_path)
                count += 1
    save_random_images(os.path.join(data_root_dir, f'Galaxy_Images/{class_name}'), class_name, data_root_dir)
    print('Number of images copied for class', class_name, ':', count, 'in', os.path.join(data_root_dir, f'Galaxy_Images/{class_name}'))

def my_accuracy_plot_formatting(ax, axis_label, xlim=None, ylim=None):    

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(left=0)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlabel(axis_label)

def distance_bin_plots(subset, xlimits=None, ylimits=None, method=None):

    # X AXIS - DISTANCE INFO
    # Extract distance columns as a numpy array
    if method == "BGMM":
        prefix = 'Prob'
        x_axis_label = 'Assigned Cluster Gaussian Density'
        width = 0.01
    elif method == "MBKM":
        prefix = 'Distance'
        x_axis_label = 'Distance to Assigned Cluster Centroid'
        width = 1

    # Create Bins
    bin_edges = np.arange(0, subset[f'Assigned_{prefix}'].max()+width, width)
    
    # Create Distance Bin Column
    subset[f'{prefix}_Bin'] = pd.cut(subset[f'Assigned_{prefix}'],bins=bin_edges,include_lowest=True)
    # Create Distance Groups
    grouped = subset.groupby(f'{prefix}_Bin', observed=False)

    # Y AXIS - NUMBER OF GALAXIES
    # Number of galaxies in each distance bin
    counts = grouped.size()
    # Finding bin centers for plotting
    centres = np.array([i.mid for i in counts.index])

    # Y AXIS - ACCURACY MEAN PER BIN
    subset["Correct"] = (subset["Predicted_Label"] == subset["Volunteer_Label"])
    accuracy = grouped["Correct"].mean()

    # Y AXIS - USER CONFIDENCE
    grouped_votes = grouped["smooth-or-featured_total-votes"].mean()

    # MAKING INDIVIDUAL PLOTS
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15,12), constrained_layout=True)

    ax1.scatter(centres, counts.values, color='sandybrown')
    my_accuracy_plot_formatting(ax1, axis_label=x_axis_label, xlim= xlimits, ylim= ylimits)
    ax1.set_ylabel("Number of Galaxies")
    ax1.grid(alpha=0.3)

    ax2.scatter(centres, accuracy.values, color='olivedrab')
    my_accuracy_plot_formatting(ax2, axis_label=x_axis_label, xlim=xlimits, ylim=ylimits)
    ax2.set_ylabel("Avg. Classification Accuracy")
    ax2.grid(alpha=0.3)

    ax3.scatter(centres, grouped_votes.values, color='mediumvioletred')
    my_accuracy_plot_formatting(ax3, axis_label=x_axis_label, xlim=xlimits, ylim=ylimits)
    ax3.set_ylabel("User Confidence: Avg. Number of Votes in Q1")
    ax3.grid(alpha=0.3)

    # Left Y-Axis: Accuracy
    ax4.scatter(centres, accuracy.values, color="olivedrab", label="Accuracy")
    ax4.set_xlabel(x_axis_label)
    ax4.set_ylabel("Avg. Classification Accuracy", color="olivedrab")
    ax4.tick_params(axis='y', labelcolor="olivedrab")
    ax4.grid(alpha=0.3, color='olivedrab')
    my_accuracy_plot_formatting(ax4, axis_label=x_axis_label, xlim=xlimits, ylim=ylimits)
    
    # Right Y-Axis: User Confidence
    ax5 = ax4.twinx()
    ax5.scatter(centres, grouped_votes.values, color="mediumvioletred")
    my_accuracy_plot_formatting(ax5, axis_label=x_axis_label, xlim=None, ylim=ylimits)
    ax5.set_ylabel("User Confidence: Avg. Number of Votes in Q1", color="mediumvioletred")
    ax5.tick_params(axis='y', labelcolor="mediumvioletred")
    ax5.grid(alpha=0.3, color='mediumvioletred')

    plt.savefig(os.path.join('Data/Thesis Plots', f'{method}_distance_bin_plots.png'), bbox_inches='tight')
    display(fig)
    plt.close(fig)

def confidence_bin_plots(subset, width, xlimits=None, ylimits=None, method=None):

    # MAKING ACCURACY VS USER CONFIDENCE PLOT
    # Create Bins
    bin_edges = np.arange(0, subset['smooth-or-featured_total-votes'].max()+width, width)
    # Create Distance Bin Column
    subset['Confidence_Bin'] = pd.cut(subset['smooth-or-featured_total-votes'],bins=bin_edges,include_lowest=True)
    # Create Distance Groups
    grouped = subset.groupby("Confidence_Bin", observed=False)
    # Number of galaxies in each distance bin
    counts = grouped.size()
    # Finding bin centers for plotting
    centres = np.array([i.mid for i in counts.index])
    accuracy = grouped["Correct"].mean()

    fig, ax1 = plt.subplots(1,1,figsize=(8,6), constrained_layout=True)
    
    ax1.scatter(centres, accuracy.values, color="olivedrab")
    ax1.set_ylabel("Avg. Classification Accuracy")
    ax1.grid(alpha=0.3)
    my_accuracy_plot_formatting(ax1, "User Confidence: Avg. Number of Votes in Q1", xlim=xlimits, ylim=ylimits)
        
    plt.savefig(os.path.join('Data/Thesis Plots', f'{method}_confidence_bin_plot.png'), bbox_inches='tight')
    display(fig)
    plt.close(fig)
    
def add_thumbnail_panel(fig, selected_indices, cluster, position, border_color, image_paths):

    panel_ax = fig.add_axes(position)
    panel_ax.set_zorder(10)
    panel_ax.set_xlim(0, 3)
    panel_ax.set_ylim(0, 3)
    # Hide ticks
    panel_ax.set_xticks([])
    panel_ax.set_yticks([])

    # Coloured border
    panel_ax.patch.set_facecolor("white")
    panel_ax.patch.set_edgecolor(border_color)
    panel_ax.patch.set_linewidth(30)

    # Display images
    for i, idx in enumerate(selected_indices):

        image_id = str(idx)

        if image_id not in image_paths:
            continue

        img = mpimg.imread(image_paths[image_id])
        imagebox = OffsetImage(img, zoom=0.12)

        row = i // 3
        col = i % 3
        x = col + 0.5
        y = 2.5 - row

        ab = AnnotationBbox(
            imagebox,
            (x, y),
            frameon=False,
            pad=0
        )

        panel_ax.add_artist(ab)

    # Cluster number above thumbnail
    panel_ax.text(0.5, 1.15, cluster,
                  transform=panel_ax.transAxes, ha="center", va="bottom",
                  fontsize=18, fontweight="bold")

    return panel_ax