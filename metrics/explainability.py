import pandas as pd
import posteriors
import numpy as np
from torch import func
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import sys
from tqdm import tqdm
from captum.attr import IntegratedGradients
from scipy import stats

sys.path.insert(0, 'metrics')
from accuracy import classification_metrics


def save_embeddings(model, model_state, model_layer_name, X_data, X_long_data, data_ids, args):
    """Helper function for saving the embeddings from the neural network

    Creates two embeddings files: one created from final layer of neural net, another from the final LSTM layer

    Parameters:
    model: the torch model to create embeddings from
    model_state: the state of the model 
    model_layer_name: a name for the model to use for saving the output (e.g., global, auxiliary_task_XXX, local_task_XXX)
    X_data: the (tabular) data to create embeddings for
    X_long_data: the (longitudinal) data to create embeddings for
    data_ids: the patient IDs (in the order the data is given in X_data, X_long_data)
    args: the args dictionary
    """
    if args['data_type'] == 'sequence':
        embeddings_final_layer = []
        embeddings_lstm_layer = []

        # average embeddings over multiple samples of BNN parameters
        for _ in range(args['num_mc_samples']):
            sample = posteriors.vi.diag.sample(model_state)
            _, final_layer_embedding, lstm_layer_embedding = func.functional_call(model, sample, (X_data, X_long_data, True))
            final_layer_embedding = final_layer_embedding.cpu().detach().numpy()
            lstm_layer_embedding = lstm_layer_embedding.cpu().detach().numpy()
            embeddings_final_layer.append(final_layer_embedding)
            embeddings_lstm_layer.append(lstm_layer_embedding)
        # format the outputs in a pandas dataframe
        embeddings_final_layer_df = pd.DataFrame(np.mean(embeddings_final_layer, axis=0))
        embeddings_final_layer_df['patient_id'] = data_ids
        embeddings_lstm_layer_df = pd.DataFrame(np.mean(embeddings_lstm_layer, axis=0))
        embeddings_lstm_layer_df['patient_id'] = data_ids
        
        # save output
        embeddings_final_layer_df.columns = embeddings_final_layer_df.columns.astype(str)
        embeddings_final_layer_df = pa.Table.from_pandas(embeddings_final_layer_df)
        pq.write_table(embeddings_final_layer_df, '{}_{}_embeddings_final_layer.parquet'.format(args['outprefix'], model_layer_name))
        embeddings_lstm_layer_df.columns = embeddings_lstm_layer_df.columns.astype(str)
        embeddings_lstm_layer_df = pa.Table.from_pandas(embeddings_lstm_layer_df)
        pq.write_table(embeddings_lstm_layer_df, '{}_{}_embeddings_lstm_layer.parquet'.format(args['outprefix'], model_layer_name))

    elif args['data_type'] == 'tabular':

        embeddings_final_layer = []

        # average embeddings over multiple samples of BNN parameters
        for _ in range(args['num_mc_samples']):
            sample = posteriors.vi.diag.sample(model_state)
            _, final_layer_embedding = func.functional_call(model, sample, (X_data, X_long_data, True))
            final_layer_embedding = final_layer_embedding.cpu().detach().numpy()
            embeddings_final_layer.append(final_layer_embedding)
        # format the outputs in a pandas dataframe
        embeddings_final_layer_df = pd.DataFrame(np.mean(embeddings_final_layer, axis=0))
        embeddings_final_layer_df['patient_id'] = data_ids
        
        # save output
        embeddings_final_layer_df.columns = embeddings_final_layer_df.columns.astype(str)
        embeddings_final_layer_df = pa.Table.from_pandas(embeddings_final_layer_df)
        pq.write_table(embeddings_final_layer_df, '{}_{}_embeddings_final_layer.parquet'.format(args['outprefix'], model_layer_name))


def get_integrated_gradients(model, X_tab, X_long, batch_size=2000):
    ig = IntegratedGradients(model)

    n_samples = X_tab.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    print("Computing feature importance for {} batches".format(n_batches))

    tab_attributions = []
    long_attributions = []

    for i in tqdm(range(n_batches), desc="Computing attributions"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        X_tab_batch = X_tab[start_idx:end_idx]
        X_long_batch = X_long[start_idx:end_idx]
        
        input_data = (X_tab_batch, X_long_batch)
        
        attributions_batch = ig.attribute(input_data, target=0, return_convergence_delta=False)
        
        tab_attributions.append(attributions_batch[0].cpu())
        long_attributions.append(attributions_batch[1].cpu())
        
        # clear CUDA cache
        torch.cuda.empty_cache()

    tab_attributions = torch.cat(tab_attributions, dim=0)
    long_attributions = torch.cat(long_attributions, dim=0)

    return tab_attributions, long_attributions


def get_ci(data, confidence=0.95):
    mean = data.mean(dim=0)
    std_error = data.std(dim=0) / np.sqrt(data.shape[0])
    df = data.shape[0] - 1
    t_value = stats.t.ppf((1 + confidence) / 2, df)
    margin_of_error = t_value * std_error
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    return mean, ci_lower, ci_upper


def integrated_gradients_importance(model, model_layer_name, X_tab, X_long, feature_names, args):
    attributions = get_integrated_gradients(model, X_tab, X_long)
    
    # separate attributions for tabular and longitudinal data
    tab_attributions, long_attributions = attributions
    
    # calculate mean absolute attribution for each feature
    tab_importance, tab_importance_lower_ci, tab_importance_upper_ci = get_ci(tab_attributions.abs())
    # sum over time dimension (dim=1) and then mean over sample dimension (dim=0)
    # this aggregates the total impact across all time steps
    if args['data_type'] == 'sequence':
        long_importance, long_importance_lower_ci, long_importance_upper_ci = get_ci(long_attributions.abs().sum(dim=1)) 
        all_importances = torch.cat([tab_importance, long_importance])
        all_importances_lower_ci = torch.cat([tab_importance_lower_ci, long_importance_lower_ci])
        all_importances_upper_ci = torch.cat([tab_importance_upper_ci, long_importance_upper_ci])
    else:
        all_importances = tab_importance
        all_importances_lower_ci = tab_importance_lower_ci
        all_importances_upper_ci = tab_importance_upper_ci
    
    # create a DataFrame with feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': all_importances.detach().numpy(),
        'importance_lower_ci': all_importances_lower_ci.detach().numpy(),
        'importance_upper_ci': all_importances_upper_ci.detach().numpy()
    })
    
    # sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    outpath = '{}_{}_feature_importance.csv'.format(args['outprefix'], model_layer_name)
    print("Saving feature importance output to {}".format(outpath))
    importance_df.to_csv(outpath, index=None)
    
    print(model_layer_name)
    print(importance_df.to_string())
    return importance_df
