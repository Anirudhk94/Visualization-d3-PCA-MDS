from flask import Flask,request,jsonify,render_template
import random
import pylab as plt
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

app = Flask(__name__)

input_df = pd.read_csv('diabetes.csv', low_memory=False)
og_df = pd.read_csv('diabetes.csv', low_memory=False)
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# print(input_file.columns.values)

random_samples = []
strat_samples = []
sample_size = 10

scaler = StandardScaler()
input_df[columns]=scaler.fit_transform(input_df[columns])

@app.route("/")
def d3():
    return render_template('index.html')

# Randomized sampling for diabetes data
@app.route("/random_sampling")
def get_random_sampling():
    # Getting the global variables
    global random_samples
    global sample_size
    global data
    global input_df

    # Fetching all the rows and converting then to numpy array
    rows = input_df[columns]
    data = np.array(rows)

    # Calling the pandas random sampling function
    random_rows = random.sample(range(len(input_df)), sample_size)
    # print(random_rows)

    # Storing the samples into the resultant list
    for row in random_rows:
        random_samples.append(data[row])
    return pd.json.dumps(random_samples)

@app.route("/stratified_sampling")
def get_stratified_sampling():
    global input_df
    global strata_samples
    global sample_size
    global strat_samples

    clusters = []
    cluster_size = []
    samples = []

    rows = input_df[columns]
    # Clustering : Creating kmeans object
    kmean_tfm = KMeans(n_clusters=4)
    # print(kmean_tfm)

    # Passing the input data for clustering
    kmean_tfm.fit(rows)
    labels = kmean_tfm.labels_
    # print(labels)

    # Adding the cluster labels to input df
    input_df['kcluster'] = pd.Series(labels)

    for i in range(4):
        clusters.append(input_df[input_df['kcluster'] == i])
        cluster_size.append(len(clusters[i]) * sample_size / len(input_df))
        samples.append(clusters[i].ix[random.sample(list(clusters[i].index), int(cluster_size[i]))])

    # print(len(input_df))
    # print(cluster_size)
    # print(samples);

    strat_samples = pd.concat([samples[0], samples[1], samples[2], samples[3]])
    # print(len(strat_samples))
    return pd.json.dumps(strat_samples)


def get_elbow_graph():
    global og_df

    # Fetch the original content of the input file
    rows = og_df[columns]

    # Set possible values of k to be in [1,8]
    k = range(1, 8)

    # Initialising kmean objects for different k values
    kmean_objects = [KMeans(n_clusters=i).fit(rows) for i in k]

    # Get the coordinates of all the cluster centers
    cluster_centers = [obj.cluster_centers_ for obj in kmean_objects]

    # Calculate the distances between all the centers and data points and select the minimum
    center_dist = [cdist(rows, center, 'euclidean') for center in cluster_centers]
    # print(center_dist)
    distances = [np.min(d_list, axis=1) for d_list in center_dist]
    # print(distances)
    avg_within = [np.sum(dist) / rows.shape[0] for dist in distances]
    # print(avg_within)

    elbow = 3
    elbow_fig = plt.figure()
    desc = elbow_fig.add_subplot(1, 1, 1)
    desc.plot(k, avg_within, 'g*-')
    desc.plot(k[elbow], avg_within[elbow], marker='o')
    plt.grid(True)
    plt.xlabel('Clusters')
    plt.ylabel('Objective Function : Average within-cluster sum of squares')
    plt.title('Elbow Curve for KMean clustering')
    plt.show()


@app.route("/scree_plot_strat")
def scree_plot_strat():
    # Plotting scree plot
    global strat_samples
    pca = PCA(n_components=8)
    pca.fit(strat_samples)
    print(pca.explained_variance_ratio_)
    variance = pca.explained_variance_ratio_
    prev_sum = 0
    variance_cdf = []
    for val in variance:
        variance_cdf.append(val + prev_sum)
        prev_sum += val
    data = {'variance':variance,
            'variance_cdf':variance_cdf}
    return pd.json.dumps(data)


@app.route("/scree_plot_random")
def scree_plot_random():
    # Plotting scree plot
    global random_samples
    pca = PCA(n_components=8)
    pca.fit(random_samples)
    print(pca.explained_variance_ratio_)
    variance = pca.explained_variance_ratio_
    prev_sum = 0
    variance_cdf = []
    for val in variance:
        variance_cdf.append(val + prev_sum)
        prev_sum += val
    data = {'variance': variance,
            'variance_cdf': variance_cdf}
    return pd.json.dumps(data)


@app.route('/pca_random')
def pca_random():
    global random_samples
    pca_data = PCA(n_components=2)
    x = random_samples
    pca_data.fit(x)
    x = pca_data.transform(x)
    data_columns = pd.DataFrame(x)
    return pd.json.dumps(data_columns)

@app.route('/pca_strat')
def pca_strat():
    global strat_samples
    pca_data = PCA(n_components=2)
    x = strat_samples
    pca_data.fit(x)
    x = pca_data.transform(x)
    data_columns = pd.DataFrame(x)
    return pd.json.dumps(data_columns)

@app.route('/mds_correlation_random')
def mds_euclidean_random():
    global random_samples
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(random_samples, metric='euclidean')
    x = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(x)
    return pd.json.dumps(data_columns)

@app.route('/mds_euclidean_strat')
def mds_euclidean_random():
    global strat_samples
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(strat_samples, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(X)
    return pd.json.dumps(data_columns)

@app.route('/mds_correlation_random')
def mds_euclidean_random():
    global random_samples
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(random_samples, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(X)
    return pd.json.dumps(data_columns)

@app.route('/mds_euclidean_strat')
def mds_euclidean_random():
    global strat_samples
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(strat_samples, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(X)
    return pd.json.dumps(data_columns)

@app.route("/")
def hello():
    return "Hello asd!"

if __name__ == "__main__":
    app.run()
    get_random_sampling()
    get_stratified_sampling()