import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import classification_report, accuracy_score

def myMBKM(features, n_clusters=20, metric_sample=None):
    # Perform fit
    my_kmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=10, max_no_improvement=None).fit(features)
    # print('K-Means Iterations: ',my_kmeans.n_iter_)
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
    distance_df = pd.DataFrame(kmeans_dist, index=clusters.index, columns=dist_cols)   
    clusters = pd.concat([clusters, distance_df], axis=1)

    return kmeans_inertia, kmeans_silhouette, kmeans_david, clusters

rng = np.random.default_rng(42)

project_dir = os.path.expanduser('~/MSc')
print('Project Directory: ', project_dir)
data_root_dir = os.path.join(project_dir, "Data")
print('Data Directory: ', data_root_dir)
output_dir = os.path.join(project_dir, "MBKM_Results")
print('Output Directory: ', output_dir)
# Complete cluster count results
temp_file = os.path.join(output_dir, "MBKM_results_temp.pkl")
final_file = os.path.join(output_dir, "MBKM_results.pkl")
# Detailed results files for 20 clusters only
detailed_temp_file = os.path.join(output_dir, "MBKM_20_1_temp.pkl")
detailed_final_file = os.path.join(output_dir, "MBKM_20_1.pkl")

# Load the prefiltered volunteer labels
volunteer_labels = pd.read_parquet(os.path.join(data_root_dir, "volunteer_labels.parquet"))
print("Volunteer-label shape:", volunteer_labels.shape, flush=True)
# Load the PCA reduced galaxy features
features = pd.read_parquet(os.path.join(data_root_dir, "pca_features.parquet"))
print("PCA Features shape:", features.shape, flush=True)
# Load sample for Silhouette and Davies-Bouldin
metric_sample = np.load(os.path.join(data_root_dir, 'metric_sample.npy'))
print("Metric Sample shape: ", len(metric_sample))
# Match galaxies present in both datasets
common_ids = features.index.intersection(volunteer_labels.index)
print("Galaxies in both datasets:", len(common_ids), flush=True)
# Prepare the evaluation subset
test_subset = volunteer_labels.loc[common_ids, ["Volunteer_Label"]].copy()
print("Evaluation subset:", test_subset.shape, flush=True)

no_of_runs = 5
no_of_clusters_to_test = np.arange(10, 101, 1)

if os.path.exists(final_file):
    print("Loading existing results...", flush=True)
    with open(final_file, "rb") as f:
        results = pickle.load(f)
    print("Existing results loaded.", flush=True)
else:
    results = {}

for no_of_clusters in no_of_clusters_to_test:
    results.setdefault(no_of_clusters, {})
    for run in range(no_of_runs):
        run_number = run + 1
        if run_number in results[no_of_clusters]:
            print(
                f"Skipping {no_of_clusters} components, "
                f"run {run_number}: already completed.",
                flush=True,
            )
            continue
        
        mbkm_inertia, silhouette, david, clusters = myMBKM(features, n_clusters=no_of_clusters, metric_sample=metric_sample)
        test_results = clusters[["Cluster"]].join(test_subset, how="inner")
        dominant_labels = test_results.groupby("Cluster")["Volunteer_Label"].agg(lambda labels: labels.value_counts().idxmax())
        # Assign the dominant volunteer-voted label to every galaxy
        clusters["Dominant_Type"] = clusters["Cluster"].map(dominant_labels)
        predicted_labels = test_results["Cluster"].map(dominant_labels)

        run_accuracy = accuracy_score(test_results["Volunteer_Label"], predicted_labels)

        results[no_of_clusters][run_number] = {
            "accuracy": run_accuracy,
            "silhouette": silhouette,
            "inertia": mbkm_inertia,
            "david_bouldin": david,
        }
        if no_of_clusters == 20:
            if run_number ==1:

                # Finding the distance/probability columns in the clusters dataframe
                distance_cols = [f"Distance_{i}" for i in range(no_of_clusters)] 
                distance_array = clusters[distance_cols].to_numpy()
                # Finding Assigned Distance by selecting the distance corresponding to the assigned cluster for each row
                clusters[f'Assigned_Dist'] = distance_array[np.arange(len(clusters)), clusters['Cluster'].to_numpy()]
                # Add predicted labels to the evaluation subset
                test_results["Predicted_Label"] = predicted_labels
                # Print the number of galaxies per class in each cluster
                class_counts = pd.crosstab(test_results["Cluster"], test_results["Volunteer_Label"])
                print("Test set class counts per cluster:")
                print(class_counts)
    
                # Calculate full cluster member counts
                full_cluster_counts = clusters["Cluster"].value_counts().sort_index()
    
                # Calculate test set cluster member counts
                test_cluster_counts = test_results["Cluster"].value_counts().sort_index()
                # Calculate individual cluster accuracies
                cluster_accuracies = {}

                for cluster in range(no_of_clusters):
                    cluster_data = test_results[test_results["Cluster"] == cluster]
                    if len(cluster_data) == 0:
                        cluster_accuracies[cluster] = np.nan
                        continue
    
                    cluster_accuracies[cluster] = accuracy_score(
                        cluster_data["Volunteer_Label"], cluster_data["Predicted_Label"])
                # Create summary DataFrame
                cluster_summary = pd.DataFrame({
                            "Cluster": range(no_of_clusters),
                            "Full_Count": [full_cluster_counts.get(i, 0) for i in range(no_of_clusters)],
                            "Test_Count": [test_cluster_counts.get(i, 0) for i in range(no_of_clusters)],
                            "Predicted_Label": [dominant_labels.get(i, None) for i in range(no_of_clusters)],
                            "Accuracy": [cluster_accuracies[i] for i in range(no_of_clusters)]
                            })
                # Print summary
                print("Cluster Summary:")
                print(cluster_summary.to_string(index=False))
    
                # Load existing detailed results if available
                if os.path.exists(detailed_final_file):
                    with open(detailed_final_file, "rb") as f:
                        detailed_results = pickle.load(f)
                else:
                    detailed_results = {}
    
                # Store the detailed results for this run
                detailed_results[run_number] = {
                    "clusters": clusters.copy(),
                    "test_subset": test_results.copy(),
                    "cluster_accuracies": cluster_accuracies,
                    "dominant_labels": dominant_labels.to_dict(),
                    "cluster_summary": cluster_summary,
                    "accuracy": run_accuracy,
                    "silhouette": silhouette,
                    "inertia": mbkm_inertia,
                    "david_bouldin": david,
                }

                # Save detailed results
                with open(detailed_temp_file, "wb") as f:
                    pickle.dump(detailed_results, f)
                os.replace(detailed_temp_file, detailed_final_file)
                print(f"Saved detailed results for 20 clusters, run {run_number}", flush=True)

            print('Classification Report\n#####################')
            print(classification_report(test_results['Volunteer_Label'], predicted_labels, labels=['R', 'S', 'E'], digits=4))
            
        print(f"{no_of_clusters} clusters, Run: {run_number}: Accuracy = {run_accuracy:.4f}", flush=True)

        with open(temp_file, "wb") as f:
            pickle.dump(results, f)
        os.replace(temp_file, final_file)

print("All MBKM runs completed.", flush=True)
print(f"Results saved to: {final_file}", flush=True)