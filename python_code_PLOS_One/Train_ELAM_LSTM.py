# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELAM-LSTM: Fish Movement Prediction using LSTM
This script trains an LSTM model to predict fish swimming behavior (u_swim, v_swim)
based on flow field features and historical movement data.

It includes:
- Data loading and preprocessing (standardization).
- Custom LSTM model definition with persistent state management.
- Training loop with Early Stopping and Learning Rate Scheduling.
- Comprehensive evaluation metrics (MSE, R2, etc.).
- Detailed plotting of results (Scatter plots, Residuals, Distributions).
"""

__author__ = "Abbas El Hachem"
__institution__ = ('Bundesanstalt für Wasserbau | BAW'
                   'Hydraulic Engineering in Inland Areas')
__copyright__ = ('GNU General Public License 3')
__version__ = 0.1
__last_update__ = '15.12.2025'

# %% define imports
import time
import timeit
import random
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr as spr
from scipy.stats import pearsonr as prs
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Reproducibility Setup ---
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device selected:', device)

# --- Utility Functions ---

def standardize_data(data: np.ndarray):
    """
    Standardizes the data based on its mean and standard deviation.
    
    Parameters:
    ----------
    data : np.ndarray or pd.DataFrame
        The input data to be standardized.
    
    Returns:
    -------
    standardized_data : np.ndarray
        Standardized data (mean=0, std=1).
    mean : np.ndarray
        Mean values of the input data for each feature.
    std : np.ndarray
        Standard deviation values of the input data for each feature.
    """
    mean = np.mean(data, axis=0)  # Calculate mean for each feature
    std = np.std(data, axis=0)    # Calculate std for each feature
    
    # Standardize the data
    standardized_data = (data - mean) / std
    
    return standardized_data, mean, std


def inverse_standardize_data(standardized_data: np.ndarray,
                             mean: np.ndarray, std: np.ndarray):
    """
    Inverse standardizes the data using the provided mean and standard deviation.
    
    Parameters:
    ----------
    standardized_data : np.ndarray
        The data that was standardized (mean=0, std=1).
    mean : np.ndarray
        Mean values of the original data.
    std : np.ndarray
        Standard deviation values of the original data.
    
    Returns:
    -------
    original_data : np.ndarray
        The inverse-standardized data, back to its original scale.
    """
    # Inverse standardize the data
    standardized_data_df = pd.DataFrame(
        index=range(len(standardized_data)),
        data=standardized_data,
        columns=std.index)
    original_data = (standardized_data_df * std) + mean
    
    return original_data.values


class FishTrackLSTM(nn.Module):
    def __init__(self, input_size, output_size, 
                 hidden_size=128, num_layers=2, dropout=0.1):
        super(FishTrackLSTM, self).__init__()
    
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer (hidden and cell states are handled automatically)
        self.lstm = nn.LSTM(input_size, hidden_size, 
                    num_layers, batch_first=True, 
                    dropout=dropout)

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.hidden_state = None
        self.cell_state = None
        
    def forward(self, src: torch.Tensor, reset_hidden: bool=False):
        """
        Forward pass with persistent hidden and cell states managed by the LSTM.
        src: (batch_size, seq_len=1, input_size) - Input data for a single time step.
        """
        batch_size, seq_len, _ = src.shape  # Get the batch_size and seq_len from input shape
        
        # If reset_hidden is True, initialize hidden and cell states
        if reset_hidden or self.hidden_state is None:
            # Initialize hidden and cell states with the correct batch size and device
            self.hidden_state = torch.zeros(self.num_layers, batch_size,
                                            self.hidden_size, device=src.device)
            self.cell_state = torch.zeros(self.num_layers, batch_size,
                                          self.hidden_size, device=src.device)

        # LSTM forward pass
        out, (hidden_state, cell_state) = self.lstm(src, (self.hidden_state, self.cell_state))
        
        # Update internal states
        self.hidden_state = hidden_state
        self.cell_state = cell_state

        # Apply dropout for regularization
        out = self.dropout(out)
        # Apply the first fully connected layer
        out = self.fc1(out)  # (batch_size, seq_len, hidden_size)
        
        # Apply dropout for regularization
        out = self.dropout_fc(out)
        
        # Apply the second fully connected layer
        out = self.fc2(out)  # (batch_size, seq_len, output_size)
        
        return out, self.hidden_state 


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(r"Early stopping triggered.") 


if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program
    
    # --- Path Definition ---
    main_proj_path = Path(r"")
    
    # Define data directories
    data_path = main_proj_path / "data"
    out_path = main_proj_path / "results"
    out_path.mkdir(parents=True, exist_ok=True)
    
    path_tracks = (data_path / r"Tracks_Versuche_A_mit_CFD.csv")

    # --- Data Loading ---
    print(f"Loading data from: {data_path}")
    if not path_tracks.exists():
        raise FileNotFoundError(f"Track file not found at {path_tracks}. Please check README for data setup.")


    # --- Hyperparameters no need to change---
    hidden_size = 300
    train_size = 0.8
    output_size = 2  # Number of outputs (u_swim, v_swim)
    num_epochs = 30
    learning_rate = 0.0001
    dropout_prob = 0.4
    num_layers = 3
    
    # input features to use (along uswim, vswim t-1)
    wanted_cols_np = [
                    #'UMean:0', 
                    #'UMean:1',
                    #'U_mag',
                    #'TKE_Mean',
                    'U:0', 
                    'U:1',
                    'U',
                    'TKE',                   
                      ]
    # ouotput features to predict (without id at t)
    wanted_cols_out = ['id', 'u_swim', 'v_swim']
    
    # Number of input features
    input_size = len(wanted_cols_np) 
    
    input_var = 'uvUk'
    save_acr = '_%s' % input_var
    print(f"Features: {wanted_cols_np}, Targets: {wanted_cols_out}")
    
    df_features_nonan = pd.read_csv(path_tracks, index_col=0, sep=';')

    print('Track flow data description:', df_features_nonan.describe())
    print('Columns:', df_features_nonan.columns)

    # get index of wanted columns
    ix_wanted_cols = [np.where(df_features_nonan.columns == _i)[0][0]
                      for _i in wanted_cols_np]
    
    data_df_out = df_features_nonan.loc[:, wanted_cols_out]
    track_ids = df_features_nonan.loc[:, 'id']

    # --- Train/Val/Test Split ---
    unique_track_ids = np.unique(track_ids)
    numb_train_tracks = round(len(unique_track_ids)*train_size)

    track_ids_train = unique_track_ids[:numb_train_tracks]
    # split for validation and test 10%-10%
    track_ids_val_test = unique_track_ids[numb_train_tracks:]
    
    numb_val_tracks = round(len(track_ids_val_test)*0.5)
    track_ids_val = track_ids_val_test[:numb_val_tracks]
    track_ids_test = track_ids_val_test[numb_val_tracks:]

    idx_train_tracks = np.concatenate(
        [np.where(df_features_nonan.iloc[:, 0] == _id)[0]
         for _id in track_ids_train])
    idx_val_tracks = np.concatenate(
        [np.where(df_features_nonan.iloc[:, 0] == _id)[0]
         for _id in track_ids_val])
    idx_test_tracks = np.concatenate(
        [np.where(df_features_nonan.iloc[:, 0] == _id)[0]
         for _id in track_ids_test])
    
    # --- Normalization ---
    normalized_input_features, mean_input, std_input = standardize_data(
        data=df_features_nonan.iloc[:, ix_wanted_cols])
    
    train_sequences = normalized_input_features.iloc[idx_train_tracks, :].copy()
    val_sequences = normalized_input_features.iloc[idx_val_tracks, :].copy()
    test_sequences = normalized_input_features.iloc[idx_test_tracks, :].copy()

    # id as columns
    train_sequences.loc[:, 'id'] = df_features_nonan.iloc[
        idx_train_tracks, np.where(df_features_nonan.columns == 'id')[0]]
    val_sequences.loc[:, 'id'] = df_features_nonan.iloc[
        idx_val_tracks, np.where(df_features_nonan.columns == 'id')[0]]
    test_sequences.loc[:, 'id'] = df_features_nonan.iloc[
        idx_test_tracks, np.where(df_features_nonan.columns == 'id')[0]]
    
    print('train_sequences columns:', train_sequences.iloc[idx_train_tracks, :].columns)
    
    # normalize output data
    normalized_output_features, mean_output, std_output = standardize_data(
        data=data_df_out.iloc[:, 1:])
    
    train_labels = normalized_output_features.iloc[idx_train_tracks, :].copy()
    val_labels = normalized_output_features.iloc[idx_val_tracks, :].copy()
    test_labels = normalized_output_features.iloc[idx_test_tracks, :].copy()
        
    # id as column
    train_labels.loc[:, 'id'] = df_features_nonan.iloc[
        idx_train_tracks, np.where(df_features_nonan.columns == 'id')[0]]
    val_labels.loc[:, 'id'] = df_features_nonan.iloc[
        idx_val_tracks, np.where(df_features_nonan.columns == 'id')[0]]
    test_labels.loc[:, 'id'] = df_features_nonan.iloc[
        idx_test_tracks, np.where(df_features_nonan.columns == 'id')[0]]
    
    print('train_labels columns:', data_df_out.iloc[idx_train_tracks, 1:].columns)
                        
    # --- Model Initialization ---
    model = FishTrackLSTM(input_size+output_size, output_size,
                          hidden_size=hidden_size, num_layers=num_layers, 
                          dropout=dropout_prob).to(device)
    
    # Initialize bias
    model.lstm.bias_hh_l0.data[hidden_size : 2 * hidden_size] = 3

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, verbose=True)

    train_loss_list = []
    val_loss_list = []
    
    # --- Training Loop ---
    model.train() 
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}/{num_epochs}')
        
        running_train_loss = 0.0
        model.train()
        
        for _id in tqdm.tqdm(track_ids_train):
            # Process one track at a time
            idx_train_track = np.where(df_features_nonan.iloc[:, 0] == _id)[0]
            track_train = train_sequences.iloc[idx_train_track, :-1]
            track_output = train_labels.iloc[idx_train_track, :-1]
            
            # Reset hidden state for new track
            model.hidden_state = None
            model.cell_state = None

            for t in range(len(track_train.index)-1):  # Iterate through time steps
                xval = track_train.iloc[t,:].values
                yval = track_output.iloc[t,:].values
                input_t = np.concatenate((xval, yval))
                
                src = torch.tensor(input_t, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
                obsv_t1 = torch.tensor(
                    track_output.iloc[t+1,:].values, 
                    dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                predicted_velocities, hidden_state = model(src, reset_hidden=False)  

                loss_uvswim = criterion(obsv_t1, predicted_velocities)
                loss_uvswim.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
    
                optimizer.step()
                
                # Detach hidden states
                model.hidden_state = model.hidden_state.detach()
                model.cell_state = model.cell_state.detach()
                
                running_train_loss += loss_uvswim.item()
            
        epoch_train_loss = running_train_loss / len(train_sequences.index)
        train_loss_list.append(epoch_train_loss)
        
        # --- Validation Loop ---
        if True:
            model.eval()
            running_val_loss = 0.0
            
            for val_id in track_ids_val:
                idx_val_track = np.where(val_sequences.iloc[:, -1] == val_id)[0]
                track_val = val_sequences.iloc[idx_val_track, :-1]
                track_output_val = val_labels.iloc[idx_val_track, :-1]
                
                # Reset states for validation track
                model.hidden_state = None
                model.cell_state = None

                for t in range(len(track_val.index)-1):
                    x_vals = track_val.iloc[t,:].values
                    y_vals = track_output_val.iloc[t,:].values
                    input_t_val = np.concatenate((x_vals, y_vals))
                    
                    src_val = torch.tensor(
                        input_t_val, 
                        dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
                    obsv_val_t1 = torch.tensor(
                        track_output_val.iloc[t+1,:].values, 
                        dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device) 
                    
                    with torch.no_grad():
                        predicted_velocities_val, _ = model(src_val)

                    loss_uvswim_val = criterion(obsv_val_t1, predicted_velocities_val)
                    running_val_loss += loss_uvswim_val.item()

            epoch_val_loss = running_val_loss / len(val_sequences)
            val_loss_list.append(epoch_val_loss)
            
            print(f'Validation Loss: {epoch_val_loss:.4f}')
    
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        
            scheduler.step(epoch_val_loss)
            
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}')
    
    # --- Save Model ---
    model.to("cpu")
    scripted_model = torch.jit.script(model)
    save_path_model = out_path / (r'lstm_seq_model_%s.pt' % save_acr)
    scripted_model.save(save_path_model)
    print(f"Model saved to {save_path_model}")
        
    # --- Evaluation on All Sets ---
    print('Training complete - Checking performance on all datasets...')
    train_predictions_all = []
    train_labels_all = []
    val_predictions_all = []
    val_labels_all = []
    test_predictions_all = []
    test_labels_all = []
    
    model.eval().to(device)
    with torch.no_grad():
        # Train data
        for _id in tqdm.tqdm(track_ids_train):
            idx_train_track = np.where(df_features_nonan.iloc[:, 0] == _id)[0]
            track_train = train_sequences.iloc[idx_train_track, :-1]
            track_output = train_labels.iloc[idx_train_track, :-1]
            
            model.hidden_state = None
            model.cell_state = None

            for t in range(len(track_train.index)-1):  
                x_vals = track_train.iloc[t,:].values
                y_vals = track_output.iloc[t,:].values
                input_t_val = np.concatenate((x_vals, y_vals))
                
                src_val = torch.tensor(
                    input_t_val, 
                    dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
                tgt = torch.tensor(
                    track_output.iloc[t+1, :].values, 
                    dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
                
                predicted_velocities,_ = model(src_val)
                train_predictions_all.append(predicted_velocities.detach().cpu().numpy())
                train_labels_all.append(tgt.detach().cpu().numpy())
                
        # Valid data
        for _id in tqdm.tqdm(track_ids_val):
            idx_valid_track = np.where(val_sequences.iloc[:, -1] == _id)[0]
            track_val = val_sequences.iloc[idx_valid_track, :-1]
            track_output = val_labels.iloc[idx_valid_track, :-1]
            
            model.hidden_state = None
            model.cell_state = None
            
            for t in range(len(track_val.index)-1):  
                x_vals = track_val.iloc[t,:].values
                y_vals = track_output.iloc[t,:].values
                input_t_val = np.concatenate((x_vals, y_vals))
                
                src_val = torch.tensor(input_t_val, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
                tgt = torch.tensor(track_output.iloc[t+1, :].values, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                
                predicted_velocities,_ = model(src_val)
                val_predictions_all.append(predicted_velocities.detach().cpu().numpy())
                val_labels_all.append(tgt.detach().cpu().numpy())
            
        # Test data
        for _id in tqdm.tqdm(track_ids_test):
            idx_test_track = np.where(test_sequences.iloc[:, -1] == _id)[0]
            track_test = test_sequences.iloc[idx_test_track, :-1]
            track_output = test_labels.iloc[idx_test_track, :-1]
            
            model.hidden_state = None
            model.cell_state = None

            for t in range(len(track_test.index)-1):  
                x_vals = track_test.iloc[t,:].values
                y_vals = track_output.iloc[t,:].values
                input_t_val = np.concatenate((x_vals, y_vals))
                
                src_val = torch.tensor(input_t_val, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
                tgt = torch.tensor(track_output.iloc[t+1, :].values, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                
                predicted_velocities,_ = model(src_val)
                test_predictions_all.append(predicted_velocities.detach().cpu().numpy())
                test_labels_all.append(tgt.detach().cpu().numpy())
            
    # --- Metrics and Processing ---
    # Concatenate all predictions and true labels from batches
    train_predictions = np.concatenate(train_predictions_all, axis=0)[:, 0]
    train_labels_plot = np.concatenate(train_labels_all, axis=0)[:, 0]
    
    val_predictions = np.concatenate(val_predictions_all, axis=0)[:, 0]
    val_labels_plot = np.concatenate(val_labels_all, axis=0)[:, 0]
    
    test_predictions = np.concatenate(test_predictions_all, axis=0)[:, 0]
    test_labels_plot = np.concatenate(test_labels_all, axis=0)[:, 0]
    
    # Inverse Normalize
    train_predictions = inverse_standardize_data(train_predictions, mean_output, std_output)
    val_predictions = inverse_standardize_data(val_predictions, mean_output, std_output)
    test_predictions = inverse_standardize_data(test_predictions, mean_output, std_output)
    
    train_labels_plot = inverse_standardize_data(train_labels_plot, mean_output, std_output)
    val_labels_plot = inverse_standardize_data(val_labels_plot, mean_output, std_output)
    test_labels_plot = inverse_standardize_data(test_labels_plot, mean_output, std_output)
        
    # Initialize DataFrame for metrics
    df_results_metrics = pd.DataFrame(columns=['metric'], 
        index=['mse_x_train', 'mse_y_train', 'r2_x_train', 'r2_y_train', 'r2_xy_train', 'r2_U_train',
               'mse_train', 'mae_train', 'mse_u_train', 'spr_train', 'prs_train',
               'r2_xy_val', 'mse_val', 'mae_val', 'r2_U_val',
               'mse_x_test', 'mse_y_test', 'r2_x_test', 'r2_y_test', 'r2_xy_test', 'r2_U_test',
               'mse_test', 'mae_test', 'mse_u_test', 'spr_test', 'prs_test'], data=0.)
                
    # --- Metric Calculations (Train) ---
    mse_x_train = mean_squared_error(train_predictions[:, 0].ravel(), train_labels_plot[:, 0].ravel())
    mse_y_train = mean_squared_error(train_predictions[:, 1].ravel(), train_labels_plot[:, 1].ravel())
    r2_x_train = r2_score(train_predictions[:, 0].ravel(), train_labels_plot[:, 0].ravel())
    r2_y_train = r2_score(train_predictions[:, 1].ravel(), train_labels_plot[:, 1].ravel())
    r2_xy_train = r2_score(train_predictions.ravel(), train_labels_plot.ravel())
    
    train_U_pred = np.sqrt(train_predictions[:, 0]**2 + train_predictions[:, 1]**2)
    train_U_true = np.sqrt(train_labels_plot[:, 0]**2 + train_labels_plot[:, 1]**2)
    r2_U_train = r2_score(train_U_pred, train_U_true)
    
    mse_train = mean_squared_error(train_predictions.ravel(), train_labels_plot.ravel())
    mae_train = mean_absolute_error(train_predictions.ravel(), train_labels_plot.ravel())
    mse_u_train = mean_squared_error(train_U_pred, train_U_true)
    spr_train = spr(train_predictions.ravel(), train_labels_plot.ravel())[0]
    prs_train = prs(train_predictions.ravel(), train_labels_plot.ravel())[0]
    
    print(f"Train MSE: {100*mse_train } [cm/s]")
    print(f"Train MAE: {100*mae_train } [cm/s]")
    
    # --- Metric Calculations (Val) ---
    val_U_pred = np.sqrt(val_predictions[:, 0]**2 + val_predictions[:, 1]**2)
    val_U_true = np.sqrt(val_labels_plot[:, 0]**2 + val_labels_plot[:, 1]**2)
    
    r2_xy_val = r2_score(val_labels_plot.ravel(), val_predictions.ravel())
    r2_U_val = r2_score(val_U_pred, val_U_true)
    mse_val = mean_squared_error(val_labels_plot.ravel(), val_predictions.ravel())
    mae_val = mean_absolute_error(val_labels_plot.ravel(), val_predictions.ravel())
    
    print(f"Val MSE: {100*mse_val } [cm/s]")
    print(f"Val MAE: {100*mae_val } [cm/s]")
    
    # --- Metric Calculations (Test) ---
    mse_x_test = mean_squared_error(test_labels_plot[:, 0].ravel(), test_predictions[:, 0].ravel())
    mse_y_test = mean_squared_error(test_labels_plot[:, 1].ravel(), test_predictions[:, 1].ravel())
    r2_x_test = r2_score(test_labels_plot[:, 0].ravel(), test_predictions[:, 0].ravel())
    r2_y_test = r2_score(test_labels_plot[:, 1].ravel(), test_predictions[:, 1].ravel())
    r2_xy_test = r2_score(test_labels_plot.ravel(), test_predictions.ravel())
    
    test_U_pred = np.sqrt(test_predictions[:, 0]**2 + test_predictions[:, 1]**2)
    test_U_true = np.sqrt(test_labels_plot[:, 0]**2 + test_labels_plot[:, 1]**2)
    r2_U_test = r2_score(test_U_pred, test_U_true)
    
    mse_test = mean_squared_error(test_labels_plot.ravel(), test_predictions.ravel())
    mae_test = mean_absolute_error(test_labels_plot.ravel(), test_predictions.ravel())
    mse_u_test = mean_squared_error(test_U_true, test_U_pred)
            
    spr_test = spr(test_labels_plot.ravel(), test_predictions.ravel())[0]
    prs_test = prs(test_labels_plot.ravel(), test_predictions.ravel())[0]
    
    print(f"Test MSE: {100*mse_test} [cm/s]")
    print(f"Test MAE: {100*mae_test} [cm/s]")
    
    # Store metrics
    df_results_metrics.loc[
       ['mse_x_train', 'mse_y_train', 'r2_x_train', 'r2_y_train', 'r2_xy_train','r2_U_train',
        'mse_train', 'mae_train', 'mse_u_train', 'spr_train', 'prs_train',
        'r2_xy_val', 'mse_val', 'mae_val', 'r2_U_val',
        'mse_x_test', 'mse_y_test', 'r2_x_test', 'r2_y_test', 'r2_xy_test', 'r2_U_test', 
        'mse_test','mae_test', 'mse_u_test', 'spr_test', 'prs_test'], 'metric'] = [
                mse_x_train, mse_y_train, r2_x_train, r2_y_train, r2_xy_train, r2_U_train,
                mse_train, mae_train, mse_u_train, spr_train, prs_train,
                r2_xy_val, mse_val, mae_val, r2_U_val,
                mse_x_test, mse_y_test, r2_x_test, r2_y_test, r2_xy_test, r2_U_test,
                mse_test, mae_test, mse_u_test, spr_test, prs_test]
    
    df_results_metrics.to_csv(out_path / (r'metrics_train_test_lstm_seq_%s.csv' % save_acr), float_format='%0.4f')   
    
    # --- Plotting Results ---
    print("Generating plots...")
    plt.ioff()
    fig, ((ax1, ax2, ax3),
          (ax4, ax5, ax6))= plt.subplots(2, 3, figsize=(12, 8), dpi=300, sharex=False, sharey=False)

    # Reference lines
    for ax in [ax1, ax2, ax4, ax5]:
        ax.vlines(0, ymin=-3, ymax=3, color='k', alpha=0.25)
        ax.hlines(0, xmin=-3, xmax=3, color='k', alpha=0.25)

    # (a) Train U_x
    ax1.scatter(train_labels_plot[1:, 0], train_predictions[1:, 0], alpha=0.5, marker='o', edgecolor='k', facecolor='lime')
    ax1.text(0.05, 0.95, '(a) Train - %s' % r'$R^{2} =%0.2f$' % (r2_x_train),
             transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax1.plot([np.min(train_labels_plot[:, 0])-0.01, np.max(train_labels_plot[:, 0])+0.01],
             [np.min(train_labels_plot[:, 0])-0.01, np.max(train_labels_plot[:, 0])+0.01],
             alpha=0.5, color='k', linestyle='-.')
    
    # (b) Train U_y
    ax2.scatter(train_labels_plot[1:, 1], train_predictions[1:, 1], marker='o', edgecolor='k', facecolor='r', alpha=0.5)
    ax2.text(0.05, 0.95, '(b) Train - %s' % r'$R^{2} =%0.2f$' % (r2_y_train), 
             transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax2.plot([np.min(train_labels_plot[:, 1])-0.01, np.max(train_labels_plot[:, 1])+0.01],
             [np.min(train_labels_plot[:, 1])-0.01, np.max(train_labels_plot[:, 1])+0.01],
             alpha=0.5, color='k', linestyle='-.')

    # (c) Test U_x
    ax4.scatter(test_labels_plot[:, 0], test_predictions[:, 0], alpha=0.5, marker='o', edgecolor='k', facecolor='lime')
    ax4.text(0.05, 0.95, '(c) Test - %s' % r'$R^{2} =%0.2f$' % (r2_x_test),
             transform=ax4.transAxes, verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax4.plot([np.min(test_labels_plot[:, 0])-0.01, np.max(test_labels_plot[:, 0])+0.01],
             [np.min(test_labels_plot[:, 0])-0.01, np.max(test_labels_plot[:, 0])+0.01],
             alpha=0.5, color='k', linestyle='-.')

    # (d) Test U_y
    ax5.scatter(test_labels_plot[:, 1], test_predictions[:, 1], marker='o', edgecolor='k', facecolor='r', alpha=0.5)
    ax5.plot([np.min(test_labels_plot[:, 1])-0.01, np.max(test_labels_plot[:, 1])+0.01],
             [np.min(test_labels_plot[:, 1])-0.01, np.max(test_labels_plot[:, 1])+0.01],
             alpha=0.5, color='k', linestyle='-.')
    ax5.text(0.05, 0.95, '(d) Test - %s' % r'$R^{2} =%0.2f$' % (r2_y_test),
             transform=ax5.transAxes, verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    # Labels
    ax1.set_ylabel(r'Model %s m %s' % ('u$_{swim}$', '$s^{-1}$'))
    ax1.set_xlabel('Observed %s m %s' % ('u$_{swim}$', '$s^{-1}$'))
    ax2.set_xlabel('Observed %s m %s' % ('v$_{swim}$', '$s^{-1}$'))
    ax2.set_ylabel('Model %s m %s' % ('v$_{swim}$', '$s^{-1}$'))
    ax4.set_ylabel(r'Model %s m %s' % ('u$_{swim}$', '$s^{-1}$'))
    ax4.set_xlabel('Observed %s m %s' % ('u$_{swim}$', '$s^{-1}$'))
    ax5.set_xlabel('Observed %s m %s' % ('v$_{swim}$', '$s^{-1}$'))
    ax5.set_ylabel('Model %s m %s' % ('v$_{swim}$', '$s^{-1}$'))

    # Grid settings
    for ax in [ax1, ax2, ax4, ax5, ax3, ax6]:
        ax.grid(alpha=0.5)

    # (e) Train Magnitude
    ax3.plot([0, 2], [0, 2], 'k-.', alpha=0.2)
    ax3.scatter(train_U_true, train_U_pred, marker='o', edgecolor='b', facecolor='c', alpha=0.5)
    ax3.text(0.05, 0.95, '(e) Train - %s' % (r'$R^{2} =%0.2f$' % (r2_U_train)),
             transform=ax3.transAxes, verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax3.plot([0, 3], [0, 3], alpha=0.5, color='k', linestyle='-.')
    
    # (f) Test Magnitude
    ax6.plot([0, 2.5], [0, 2.5], 'k-.', alpha=0.5)
    ax6.scatter(test_U_true, test_U_pred, marker='o', edgecolor='b', facecolor='c', alpha=0.5)
    ax6.text(0.05, 0.95, '(f) Test - %s' % (r'$R^{2} =%0.2f$' % (r2_U_test)),
             transform=ax6.transAxes, verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    ax3.set_xlabel('Observed %s m %s' % ('U$_{swim}$', '$s^{-1}$'))
    ax6.set_xlabel('Observed %s m %s' % ('U$_{swim}$', '$s^{-1}$'))
    ax6.set_ylabel('Model %s m %s' % ('U$_{swim}$', '$s^{-1}$'))
    ax3.set_ylabel('Model %s m %s' % ('U$_{swim}$', '$s^{-1}$'))

    plt.tight_layout()
    plt.savefig(out_path / (r'U_uv_train_test_lstm_seq_%s.png' % save_acr), bbox_inches='tight')
    plt.close()

    # --- Plot Loss ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=300, sharex=True)
    ax1.plot(train_loss_list, label='Train Loss', color='r', marker='o')
    ax2.plot(val_loss_list, label='Validation Loss', color='b', marker='o')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc=0)
    ax1.grid(True)
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc=0)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(out_path / (r'Loss_lstm_seq_%s.png' % save_acr), bbox_inches='tight')
    plt.close()

    print(f"Done! Results saved in {out_path}")
