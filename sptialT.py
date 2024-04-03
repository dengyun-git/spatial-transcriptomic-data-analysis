# This script is used to process spatial transcriptomic data. This is not my original work, but with toy data, aiming to curate a complete workflow for single cell spatial transcriptomic data analysis.
# 2024. Keep updating with accumulated knowledge.

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from matplotlib import pyplot as plt
import scanpy as sc
import squidpy as sq
import tensorflow as tf
import seaborn as sns
from anndata import AnnData

#-------------------------------
# Load  Data into AnnData Object
#-------------------------------
### read in AnnData object adata
adata = sq.read.vizgen(
    "tutorial_data",
    counts_file="Liver1Slice1_cell_by_gene.csv",
    meta_file="Liver1Slice1_cell_metadata.csv",
)

### calculate QC matrix
adata.var_names_make_unique() ### make gene name unique
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=(50, 100, 200, 300), inplace=True
)

### Filter cells with low expression and genes that are expressed in too few cells.
sc.pp.filter_cells(adata, min_counts=50)
sc.pp.filter_genes(adata, min_cells=10)

#---------------------
# Data Pre-processing
#---------------------
# total-count normalize, logarithmize, and scale gene expression to unit variance
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# ------------------------
# Dimensionality Reduction
# ------------------------
resolution = 1.5
sc.tl.pca(adata, svd_solver="arpack") # reduce the dimensionality by PCA
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20) # neighborhood graph of cells in PCA space
sc.tl.umap(adata) # display cells on UMAP
sc.tl.leiden(adata, resolution=resolution) # clusters of cells using Leiden clustering

sc.set_figure_params(figsize=(10, 10))
sc.pl.umap(adata, color=["leiden"], size=5)

# -------------------------------
# Spatial Distributions of Cells
# -------------------------------
sq.pl.spatial_scatter(
    adata, shape=None, color="leiden", size=0.5, library_id="spatial", figsize=(10, 10)
)

# -----------------
# Assign Cell Types
# -----------------
### Reference Cell Type Marker Gene Sets
gene_panel = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41421-021-00266-1/MediaObjects/41421_2021_266_MOESM1_ESM.xlsx"
df_ref_panel_ini = pd.read_excel(gene_panel, index_col=0)
df_ref_panel = df_ref_panel_ini.iloc[1:, :1]
df_ref_panel.index.name = None
df_ref_panel.columns = ["Function"]

# Assign marker gene metadata using reference dataset
marker_genes = df_ref_panel[
    df_ref_panel["Function"].str.contains("marker")
].index.tolist()

meta_gene = deepcopy(adata.var)
common_marker_genes = list(set(meta_gene.index.tolist()).intersection(marker_genes))
meta_gene.loc[common_marker_genes, "Markers"] = df_ref_panel.loc[
    common_marker_genes, "Function"
]
meta_gene["Markers"] = meta_gene["Markers"].apply(
    lambda x: "N.A." if "marker" not in str(x) else x
)
meta_gene["Markers"].value_counts()

### Calculate Leiden Cluster Average Expression Signatures
ser_counts = adata.obs["leiden"].value_counts()
ser_counts.name = "cell counts"
meta_leiden = pd.DataFrame(ser_counts)

cat_name = "leiden"
sig_leiden = pd.DataFrame(
    columns=adata.var_names, index=adata.obs[cat_name].cat.categories
)
for clust in adata.obs[cat_name].cat.categories:
    sig_leiden.loc[clust] = adata[adata.obs[cat_name].isin([clust]), :].X.mean(0)
sig_leiden = sig_leiden.transpose()
leiden_clusters = ["Leiden-" + str(x) for x in sig_leiden.columns.tolist()]
sig_leiden.columns = leiden_clusters
meta_leiden.index = sig_leiden.columns.tolist()
meta_leiden["leiden"] = pd.Series(
    meta_leiden.index.tolist(), index=meta_leiden.index.tolist()
)

### Assign Cell Type Based on Top Expressed Marker Genes
meta_gene = pd.DataFrame(index=sig_leiden.index.tolist())
meta_gene["info"] = pd.Series("", index=meta_gene.index.tolist())
meta_gene["Markers"] = pd.Series("N.A.", index=sig_leiden.index.tolist())
meta_gene.loc[common_marker_genes, "Markers"] = df_ref_panel.loc[
    common_marker_genes, "Function"
]

meta_leiden["Cell_Type"] = pd.Series("N.A.", index=meta_leiden.index.tolist())
num_top_genes = 30
for inst_cluster in sig_leiden.columns.tolist():
    top_genes = (
        sig_leiden[inst_cluster]
        .sort_values(ascending=False)
        .index.tolist()[:num_top_genes]
    )

    inst_ser = meta_gene.loc[top_genes, "Markers"]
    inst_ser = inst_ser[inst_ser != "N.A."]
    ser_counts = inst_ser.value_counts()

    max_count = ser_counts.max()

    max_cat = "_".join(sorted(ser_counts[ser_counts == max_count].index.tolist()))
    max_cat = max_cat.replace(" marker", "").replace(" ", "-")

    print(inst_cluster, max_cat)
    meta_leiden.loc[inst_cluster, "Cell_Type"] = max_cat

# rename clusters
meta_leiden["name"] = meta_leiden.apply(
    lambda x: x["Cell_Type"] + "_" + x["leiden"], axis=1
)
leiden_names = meta_leiden["name"].values.tolist()
meta_leiden.index = leiden_names

# transfer cell type labels to single cells
leiden_to_cell_type = deepcopy(meta_leiden)
leiden_to_cell_type.set_index("leiden", inplace=True)
leiden_to_cell_type.index.name = None

adata.obs["Cell_Type"] = adata.obs["leiden"].apply(
    lambda x: leiden_to_cell_type.loc["Leiden-" + str(x), "Cell_Type"]
)
adata.obs["Cluster"] = adata.obs["leiden"].apply(
    lambda x: leiden_to_cell_type.loc["Leiden-" + str(x), "name"]
)

### Central and Portal Blood Vessels
sq.pl.spatial_scatter(
    adata, color=["Vwf", "Axin2"], size=15, cmap="Reds", img=False, figsize=(12, 8)
)

### Distinguishing Peri-Portal and Peri-Central Hepatocytes
all_hepatocyte_clusters = [x for x in meta_leiden.index.tolist() if "Hepatocyte" in x]
sig_leiden.columns = meta_leiden.index.tolist()
ser_axin2 = sig_leiden[all_hepatocyte_clusters].loc["Axin2"]
peri_central = ser_axin2[ser_axin2 > 0].index.tolist()
peri_portal = ser_axin2[ser_axin2 <= 0].index.tolist()

### Peri-Central Hepatocytes
sq.pl.spatial_scatter(
    adata, groups=peri_central, color="Cluster", size=15, img=False, figsize=(15, 15)
)

# -----------------------
# Neighborhood Enrichment
# -----------------------
sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial")
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
sq.pl.nhood_enrichment(
    adata,
    cluster_key="leiden",
    method="average",
    cmap="inferno",
    vmin=-50,
    vmax=100,
    figsize=(5, 5),
)

# --------------------------------
# Neighborhood Enrichment Clusters
# --------------------------------
n_clusters = [4]
df_nhood_enr = pd.DataFrame(
    adata.uns["leiden_nhood_enrichment"]["zscore"],
    columns=leiden_clusters,
    index=leiden_clusters,
)
nhood_cluster_levels = ["Level-" + str(x) for x in n_clusters]
linkage = sch.linkage(df_nhood_enr, method="average")
mat_nhood_clusters = sch.cut_tree(linkage, n_clusters=n_clusters)
df_cluster = pd.DataFrame(
    mat_nhood_clusters, columns=nhood_cluster_levels, index=meta_leiden.index.tolist()
)

inst_level = "Level-" + str(n_clusters[0])
all_clusters = list(df_cluster[inst_level].unique())
# sc.set_figure_params(figsize=(10,10))
for inst_cluster in all_clusters:
    inst_clusters = df_cluster[df_cluster[inst_level] == inst_cluster].index.tolist()

    sq.pl.spatial_scatter(
        adata,
        groups=inst_clusters,
        color="Cluster",
        size=15,
        img=False,
        figsize=(15, 15),
    )

# --------------------------
# Network Centrality Scores
# --------------------------
sq.gr.centrality_scores(adata, "leiden")
sc.set_figure_params(figsize=(20, 8))

# copy centrality data to new DataFrame
df_central = deepcopy(adata.uns["leiden_centrality_scores"])
df_central.index = meta_leiden.index.tolist()

# sort clusters based on centrality scores
################################################
# closeness centrality - measure of how close the group is to other nodes.
ser_closeness = df_central["closeness_centrality"].sort_values(ascending=False)

# degree centrality - fraction of non-group members connected to group members.
# [Networkx](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.degree_centrality.html#networkx.algorithms.centrality.degree_centrality)
# The degree centrality for a node v is the fraction of nodes it is connected to.
ser_degree = df_central["degree_centrality"].sort_values(ascending=False)

# clustering coefficient - measure of the degree to which nodes cluster together.
ser_cluster = df_central["average_clustering"].sort_values(ascending=False)

# --------------------
# High Closeness Score
# --------------------
inst_clusters = ser_closeness.index.tolist()[:5]
print(inst_clusters)
sq.pl.spatial_scatter(
    adata, groups=inst_clusters, color="Cluster", size=15, img=False, figsize=(10, 10)
)

### Low Closeness Score
inst_clusters = ser_closeness.index.tolist()[-5:]
print(inst_clusters)
sq.pl.spatial_scatter(
    adata, groups=inst_clusters, color="Cluster", size=15, img=False, figsize=(10, 10)
)

# -----------------------
# High Degree Centrality
# -----------------------
inst_clusters = ser_degree.index.tolist()[:5]
print(inst_clusters)
sq.pl.spatial_scatter(
    adata, groups=inst_clusters, color="Cluster", size=15, img=False, figsize=(10, 10)
)

### Low Degree Centrality
nst_clusters = ser_degree.index.tolist()[-5:]
print(inst_clusters)
sq.pl.spatial_scatter(
    adata, groups=inst_clusters, color="Cluster", size=15, img=False, figsize=(10, 10)
)

# ---------------------------
# High Clustering Coefficient
# ---------------------------
inst_clusters = ser_cluster.index.tolist()[:5]
print(inst_clusters)
sq.pl.spatial_scatter(
    adata, groups=inst_clusters, color="Cluster", size=15, img=False, figsize=(10, 10)
)

### Low Clustering Coefficient
inst_clusters = ser_cluster.index.tolist()[-5:]
print(inst_clusters)
sq.pl.spatial_scatter(
    adata, groups=inst_clusters, color="Cluster", size=15, img=False, figsize=(15, 15)
)

# --------------------------------
# Autocorrelation: Moran’s I Score
# --------------------------------
sq.gr.spatial_autocorr(adata, mode="moran")
num_view = 12
top_autocorr = (
    adata.uns["moranI"]["I"].sort_values(ascending=False).head(num_view).index.tolist()
)
bot_autocorr = (
    adata.uns["moranI"]["I"].sort_values(ascending=True).head(num_view).index.tolist()
)

### Genes with high spatial autocorrelation
sq.pl.spatial_scatter(
    adata, color=top_autocorr, size=20, cmap="Reds", img=False, figsize=(5, 5)
)
# top cell types based on average expression of top_autocorr genes
sig_leiden.loc[top_autocorr].mean(axis=0).sort_values(ascending=False).index.tolist()[
    :5
]

### Genes with low autocorrelation
sq.pl.spatial_scatter(
    adata, color=bot_autocorr, size=20, cmap="Reds", img=False, figsize=(5, 5)
)

# top cell types based on average expression of bot_autocorr genes
sig_leiden.loc[bot_autocorr].mean(axis=0).sort_values(ascending=False).index.tolist()[
    :5
]

# ----------------------------------------------
# Predict cluster labels spots using Tensorflow
# ----------------------------------------------
### squidpy.im.ImageContainer work with modern deep learning frameworks (Tensorflow)
### leverage ResNet model to generate a new set of features providing useful insights on spots similarity based on image morphology
adata = sq.datasets.visium_hne_adata()
img = sq.datasets.visium_hne_image()

# get train,test split stratified by cluster labels
train_idx, test_idx = train_test_split(
    adata.obs_names.values,
    test_size=0.2,
    stratify=adata.obs["cluster"],
    shuffle=True,
    random_state=42,
)

def get_ohe(adata: AnnData, cluster_key: str, obs_names: np.ndarray):
    cluster_labels = adata[obs_names, :].obs["cluster"]
    classes = cluster_labels.unique().shape[0]
    cluster_map = {v: i for i, v in enumerate(cluster_labels.cat.categories.values)}
    labels = np.array([cluster_map[c] for c in cluster_labels], dtype=np.uint8)
    labels_ohe = tf.one_hot(labels, depth=classes, dtype=tf.float32)
    return labels_ohe


def create_dataset(
    adata: AnnData,
    img: ImageContainer,
    obs_names: np.ndarray,
    cluster_key: str,
    augment: bool,
    shuffle: bool,
):
    # image dataset
    spot_generator = img.generate_spot_crops(
        adata,
        obs_names=obs_names,  # this arguent specified the observations names
        scale=1.5,  # this argument specifies that we will consider some additional context under each spot. Scale=1 would crop the spot with exact coordinates
        as_array="image",  # this line specifies that we will crop from the "image" layer. You can specify multiple layers to obtain crops from multiple pre-processing steps.
        return_obs=False,
    )
    image_dataset = tf.data.Dataset.from_tensor_slices([x for x in spot_generator])

    # label dataset
    lab = get_ohe(adata, cluster_key, obs_names)
    lab_dataset = tf.data.Dataset.from_tensor_slices(lab)

    ds = tf.data.Dataset.zip((image_dataset, lab_dataset))

    if shuffle:  # if you want to shuffle the dataset during training
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
    ds = ds.batch(64)  # batch
    processing_layers = [
        preprocessing.Resizing(128, 128),
        preprocessing.Rescaling(1.0 / 255),
    ]
    augment_layers = [
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(0.8),
    ]
    if augment:  # if you want to augment the image crops during training
        processing_layers.extend(augment_layers)

    data_processing = tf.keras.Sequential(processing_layers)

    ds = ds.map(lambda x, y: (data_processing(x), y))  # add processing to dataset
    return ds

train_ds = create_dataset(adata, img, train_idx, "cluster", augment=True, shuffle=True)
test_ds = create_dataset(adata, img, test_idx, "cluster", augment=True, shuffle=True)

### use pre-trained ResNet on ImageNet, a dense layer for output
input_shape = (128, 128, 3)  # input shape
inputs = tf.keras.layers.Input(shape=input_shape)

# load Resnet with pre-trained imagenet weights
x = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=input_shape,
    classes=15,
    pooling="avg",
)(inputs)
outputs = tf.keras.layers.Dense(
    units=15,  # add output layer
)(x)
model = tf.keras.Model(inputs, outputs)  # create model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # add optimizer
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),  # add loss
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    verbose=2,
)

### Calculate embedding and visualize results
### first create a new dataset, that contains the full list of spots, in the correct order and without augmentation.
full_ds = create_dataset(
    adata, img, adata.obs_names.values, "cluster", augment=False, shuffle=False
)

### another model without the output layer, in order to get the final embedding layer.
model_embed = tf.keras.Model(inputs, x)
embedding = model_embed.predict(full_ds)

### save the embedding in a new AnnData, and copy over all the relevant metadata from the AnnData with gene expression counts
adata_resnet = AnnData(embedding, obs=adata.obs.copy())
adata_resnet.obsm["spatial"] = adata.obsm["spatial"].copy()
adata_resnet.uns = adata.uns.copy()
adata_resnet

### perform the standard clustering analysis.
sc.pp.scale(adata_resnet)
sc.pp.pca(adata_resnet)
sc.pp.neighbors(adata_resnet)
sc.tl.leiden(adata_resnet, key_added="resnet_embedding_cluster")
sc.tl.umap(adata_resnet)

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.pl.umap(
    adata_resnet, color=["cluster", "resnet_embedding_cluster"], size=100, wspace=0.7
)

sq.pl.spatial_scatter(
    adata_resnet,
    color=["cluster", "resnet_embedding_cluster"],
    frameon=False,
    wspace=0.5,
)