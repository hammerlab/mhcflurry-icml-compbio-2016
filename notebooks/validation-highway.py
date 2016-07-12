
# coding: utf-8

# In[1]:

import os

import theano
print(theano.config.device)

import mhcflurry, seaborn, numpy, pandas, pickle, sklearn, collections, scipy, time
import mhcflurry.dataset
import fancyimpute, locale

import sklearn.metrics
import sklearn.cross_validation

from keras.models import Sequential
from keras.layers.core import Dense, Highway, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization

import theano

from mhcflurry.feedforward_hyperparameters import (
    LOSS,
    OPTIMIZER,
    ACTIVATION,
    BATCH_NORMALIZATION,
    INITIALIZATION_METHOD,
    DROPOUT_PROBABILITY,
    HIDDEN_LAYER_SIZE
)
from mhcflurry.regression_target import MAX_IC50, ic50_to_regression_target


def print_full(x):
    pandas.set_option('display.max_rows', len(x))
    print(x)
    pandas.reset_option('display.max_rows')
    

# In[2]:

max_ic50 = 50000
min_peptides_to_consider_allele = 10
data_dir = "../data/"


# In[3]:

all_train_data = mhcflurry.dataset.Dataset.from_csv(data_dir + "bdata.2009.mhci.public.1.txt")


# In[5]:

imputed_train_data = all_train_data.impute_missing_values(
    fancyimpute.MICE(n_imputations=50, n_burn_in=5),
    min_observations_per_peptide=5,
    min_observations_per_allele=2,
)


# In[7]:

imputed_train_data.to_dataframe()


# In[8]:

validation_df = pandas.read_csv("../data/combined_test_BLIND_dataset_from_kim2013.csv")
validation_df


# In[16]:

validation_allele_counts = validation_df.allele.value_counts()
validation_allele_counts_dict = dict(validation_allele_counts)
train_allele_counts = all_train_data._df.allele.value_counts()
train_allele_counts_dict = dict(train_allele_counts)
for common_allele in sorted(
        set(validation_allele_counts.keys()).intersection(set(train_allele_counts.keys())),
        key=lambda x: train_allele_counts[x]):
    print("\t%s %d %d" % (
            common_allele,
            train_allele_counts_dict[common_allele],
            validation_allele_counts_dict[common_allele]))


# In[20]:

alleles = sorted(train_allele_counts.index[
    (train_allele_counts >= min_peptides_to_consider_allele)
    & (train_allele_counts.index.isin(validation_allele_counts.index))
], key=lambda allele: -1 * train_allele_counts[allele])
alleles
not_dropped = []
for allele in alleles:
    sub = all_train_data.get_allele(allele)._df
    if (sub.affinity < 500).sum() < 5 or (sub.affinity > 500).sum() < 5:
        print("Dropping allele %s" % allele)
    else:
        not_dropped.append(allele)
alleles = not_dropped
len(alleles)


# In[29]:

dropout_probabilities = [0.5]
embedding_output_dims_and_layer_sizes_list = [(32, [64, 64, 64])] # , (8, [4])]
activations = ["relu"]

models_params_list = []

for model_num in range(10):
    for negative_samples in [100]:
        for impute in [False, True]:
            for dropout_probability in dropout_probabilities:
                for (embedding_output_dim, layer_sizes) in embedding_output_dims_and_layer_sizes_list:
                    for activation in activations:
                        models_params_list.append(dict(
                            negative_samples=negative_samples,
                            impute=impute,
                            dropout_probability=dropout_probability,  
                            embedding_output_dim=embedding_output_dim,
                            layer_sizes=layer_sizes,
                            activation=activation))

print("%d models" % len(models_params_list))
models_params_explored = set.union(*[set(x) for x in models_params_list])
models_params_explored


# In[30]:

def make_scores(ic50_y, ic50_y_pred, sample_weight=None, threshold_nm=500):     
    y_pred = mhcflurry.regression_target.ic50_to_regression_target(ic50_y_pred, max_ic50)
    try:
        auc = sklearn.metrics.roc_auc_score(ic50_y <= threshold_nm, y_pred, sample_weight=sample_weight)
    except ValueError:
        auc = numpy.nan
    try:
        f1 = sklearn.metrics.f1_score(ic50_y <= threshold_nm, ic50_y_pred <= threshold_nm, sample_weight=sample_weight)
    except ValueError:
        f1 = numpy.nan
    try:
        tau = scipy.stats.kendalltau(ic50_y_pred, ic50_y)[0]
    except ValueError:
        tau = numpy.nan
    
    return dict(
        auc=auc,
        f1=f1,
        tau=tau,
    )    


# In[31]:

models_and_scores = {}
validation_df_with_mhcflurry = validation_df.copy()

def make_network(
        input_size,
        embedding_input_dim=None,
        embedding_output_dim=None,
        layer_sizes=[100],
        activation=ACTIVATION,
        init="glorot_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=True,
        initial_embedding_weights=None,
        embedding_init_method="glorot_uniform",
        model=None,
        optimizer=OPTIMIZER,
        loss=LOSS):

    if model is None:
        model = Sequential()

    if embedding_input_dim:
        if not embedding_output_dim:
            raise ValueError(
                "Both embedding_input_dim and embedding_output_dim must be set")

        if initial_embedding_weights:
            n_rows, n_cols = initial_embedding_weights.shape
            if n_rows != embedding_input_dim or n_cols != embedding_output_dim:
                raise ValueError(
                    "Wrong shape for embedding: expected (%d, %d) but got (%d, %d)" % (
                        embedding_input_dim, embedding_output_dim,
                        n_rows, n_cols))
            model.add(Embedding(
                input_dim=embedding_input_dim,
                output_dim=embedding_output_dim,
                input_length=input_size,
                weights=[initial_embedding_weights],
                dropout=dropout_probability))
        else:
            model.add(Embedding(
                input_dim=embedding_input_dim,
                output_dim=embedding_output_dim,
                input_length=input_size,
                init=embedding_init_method,
                dropout=dropout_probability))
        model.add(Flatten())

        input_size = input_size * embedding_output_dim

    layer_sizes = (input_size,) + tuple(layer_sizes)

    for i, dim in enumerate(layer_sizes):
        if i == 0:
            # input is only conceptually a layer of the network,
            # don't need to actually do anything
            continue

        previous_dim = layer_sizes[i - 1]

        # hidden layer fully connected layer
        model.add(
            Highway(init=init))
        model.add(Activation(activation))

        if batch_normalization:
            model.add(BatchNormalization())

        if dropout_probability > 0:
            model.add(Dropout(dropout_probability))

    # output
    model.add(Dense(
        input_dim=layer_sizes[-1],
        output_dim=1,
        init=init))
    model.add(Activation(output_activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def make_embedding_network(
        peptide_length=9,
        n_amino_acids=20,
        embedding_output_dim=20,
        **kwargs):
    """
    Construct a feed-forward neural network whose inputs are vectors of integer
    indices.
    """
    return make_network(
        input_size=peptide_length,
        embedding_input_dim=n_amino_acids,
        embedding_output_dim=embedding_output_dim,
        **kwargs)


def from_hyperparameters(
        name=None,
        max_ic50=MAX_IC50,
        peptide_length=9,
        n_amino_acids=20,
        allow_unknown_amino_acids=True,
        embedding_output_dim=20,
        layer_sizes=[HIDDEN_LAYER_SIZE],
        activation=ACTIVATION,
        init=INITIALIZATION_METHOD,
        output_activation="sigmoid",
        dropout_probability=DROPOUT_PROBABILITY,
        loss=LOSS,
        optimizer=OPTIMIZER,
        batch_normalization=BATCH_NORMALIZATION,
        **kwargs):
    """
    Create untrained predictor with the given hyperparameters.
    """
    model = make_embedding_network(
        peptide_length=peptide_length,
        n_amino_acids=n_amino_acids + int(allow_unknown_amino_acids),
        embedding_output_dim=embedding_output_dim,
        layer_sizes=layer_sizes,
        activation=activation,
        init=init,
        loss=loss,
        optimizer=optimizer,
        output_activation=output_activation,
        dropout_probability=dropout_probability)
    return mhcflurry.Class1BindingPredictor(
        name=name,
        max_ic50=max_ic50,
        model=model,
        allow_unknown_amino_acids=allow_unknown_amino_acids,
        kmer_size=peptide_length,
        **kwargs)


# train and test models, adding columns to validation_df_with_mhcflurry
pandas.DataFrame(models_params_list).to_csv("../data/highway2l_validation_models.csv", index=False)

def make_and_fit_model(allele, original_params):
    params = dict(original_params)
    impute = params.pop("impute")
    
    negative_samples = params.pop("negative_samples")
    
    model = from_hyperparameters(max_ic50=max_ic50, **params)
    print("Fitting model for allele %s (%d + %d): %s" % (
            allele,
            len(all_train_data.groupby_allele_dictionary()[allele]),
            len(imputed_train_data.groupby_allele_dictionary()[allele]),
            str(original_params)))
    t = -time.time()
    model.fit_dataset(
        all_train_data.get_allele(allele),
        pretraining_dataset=imputed_train_data.get_allele(allele) if impute else None,
        verbose=False,
        batch_size=128,
        n_training_epochs=250,
        n_random_negative_samples=negative_samples)
    t += time.time()
    print("Trained in %d sec" % t)
    return model

for (i, allele) in enumerate(alleles):
    if allele not in validation_df_with_mhcflurry.allele.unique():
        print("Skipping allele %s: not in test set" % allele)
        continue
    if allele in models_and_scores:
        print("Skipping allele %s: already done" % allele)
        continue
    values_for_allele = []
    for (j, params) in enumerate(models_params_list):
        print("Allele %d model %d" % (i, j))
        model = make_and_fit_model(allele, params)
        predictions = model.predict(
            list(validation_df_with_mhcflurry.ix[validation_df_with_mhcflurry.allele == allele].peptide))
        print("test set size: %d" % len(predictions))
        validation_df_with_mhcflurry.loc[(validation_df_with_mhcflurry.allele == allele),
                                         ("mhcflurry %d" % j)] = predictions
        scores = make_scores(validation_df_with_mhcflurry.ix[validation_df.allele == allele].meas,
                            predictions)
        print(scores)
        values_for_allele.append((params, scores))

    models_and_scores[allele] = values_for_allele
 
    # Write out all data after each allele.
    validation_df_with_mhcflurry_results = validation_df_with_mhcflurry.ix[validation_df_with_mhcflurry.allele.isin(models_and_scores)]
    validation_df_with_mhcflurry_results.to_csv("../data/highway2l_validation_predictions_full.csv", index=False)
 
    scores_df = collections.defaultdict(list)
    predictors = validation_df_with_mhcflurry_results.columns[4:]

    for (allele, grouped) in validation_df_with_mhcflurry_results.groupby("allele"):
        scores_df["allele"].append(allele)
        scores_df["test_size"].append(len(grouped.meas))
        for predictor in predictors:
            scores = make_scores(grouped.meas, grouped[predictor])
            for (key, value) in scores.items():
                scores_df["%s_%s" % (predictor, key)].append(value)

    scores_df = pandas.DataFrame(scores_df)
    scores_df["train_size"] = [
        len(all_train_data.groupby_allele_dictionary()[a])
        for a in scores_df.allele
    ]

    scores_df.index = scores_df.allele
    scores_df.to_csv("../data/highway2l_validation_scores.csv", index=False)
        


# In[ ]:

print_full(scores_df[["train_size", "test_size"]].sort("train_size", inplace=False))


# In[ ]:

2

