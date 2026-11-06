import os
import pandas as pd
import numpy as np

def feature_pre_processing(features, data_root_dir):

    my_pca = pca.PCA_Decomposer(force_rerun=True, n_components=29, threshold=0.95, output_dir=data_root_dir)
    pca_features = my_pca.run(features)
    print('PCA Features: ', pca_features.shape)

    return pca_features

def label_pre_processing(full_volunteer_labels, data_root_dir):

    full_volunteer_labels = full_volunteer_labels.set_index('iauname')
    print('Full GZD-5 Volunteer Labels: ', full_volunteer_labels.shape)
    full_volunteer_labels = full_volunteer_labels[~full_volunteer_labels['wrong_size_warning']]
    print('wrong_size_warning filtered GZD-5 Volunteer Labels: ', full_volunteer_labels.shape)

    my_label_df, summ = create_gz_evaluation_set(full_volunteer_labels, question_level=3, min_votes=34, max_votes=None, 
                                                 half_votes_min=True, min_prob=0, max_prob=1, min_sources=100)
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

    # Combining the indices of the three classes to create a single DataFrame of volunteer labels
    volunteer_labels_indices = ellipticals.union(spirals).union(edge_ons)
    # Filtering the full volunteer labels to only include the galaxies that belong to the three classes
    volunteer_labels = full_volunteer_labels.loc[volunteer_labels_indices]
    # Assigning class labels to the volunteer labels DataFrame
    volunteer_labels.loc[ellipticals, 'Volunteer_Label'] = 'R'   # Round ellipticals
    volunteer_labels.loc[spirals, 'Volunteer_Label'] = 'S'       # Spirals
    volunteer_labels.loc[edge_ons, 'Volunteer_Label'] = 'E'      # Edge-ons

    print('Labelled Galaxies: ', volunteer_labels.shape)
    print('Round Ellipticals: ', (volunteer_labels['Volunteer_Label'] == 'R').sum())
    print('Spirals: ',           (volunteer_labels['Volunteer_Label'] == 'S').sum())
    print('Edge-ons: ',          (volunteer_labels['Volunteer_Label'] == 'E').sum())

    return volunteer_labels

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

project_dir = os.path.expanduser('~/MSc')
print('Project Directory: ', project_dir)
data_root_dir = os.path.join(project_dir, "Data")
print('Data Directory: ', data_root_dir)
label_dir = os.path.join(data_root_dir, 'gz_decals_volunteers_5.parquet')
print('Label Directory: ', label_dir)
feature_dir = os.path.join(data_root_dir, 'GalaxyZoo_Main_Features.parquet')
print('Feature Firectory: ', feature_dir)

# Load the features and labels
features = pd.read_parquet(feature_dir)
full_volunteer_labels = pd.read_parquet(label_dir)

pca_features = feature_pre_processing(features, data_root_dir)
pca_features.to_parquet(os.path.join(data_root_dir, 'pca_features.parquet'))
print("Features have been PCA reduced", flush=True)
print("PCA features are saved to: Data/pca_features.parquet", flush=True)

volunteer_labels = label_pre_processing(full_volunteer_labels, data_root_dir)
volunteer_labels.to_parquet(os.path.join(data_root_dir, 'volunteer_labels.parquet'))
print("Volunteer labels have been filtered", flush=True)
print("Filtered labels are saved to: Data/volunteer_labels.parquet", flush=True)

rng = np.random.default_rng(42)
metric_sample = rng.choice(len(pca_features), size=10000, replace=False)
np.save('metric_sample.npy', metric_sample)
print("Random sample of 10 000 galaxies has been extracted", flush=True)
print("Metric Sample saved to: Data/pmetric_sample.npy", flush=True)