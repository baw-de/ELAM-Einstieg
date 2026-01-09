# ELAM-LSTM: Fish Movement Prediction

This repository contains the implementation of **ELAM-LSTM**, a deep learning model designed to predict fish swimming components ($u_{swim}$, $v_{swim}$) based on flow field features and historical movement data.

## Overview

The model uses a Long Short-Term Memory (LSTM) network to process sequential data from fish tracks. It includes:
* **Custom LSTM Architecture:** Manages persistent hidden/cell states for continuous track prediction.
* **Physics-Informed Data:** Integrates Computational Fluid Dynamics (CFD) flow field data.
* **Robust Training:** Implements early stopping, learning rate scheduling, and dropout for regularization.

## Project Structure

```text
├── Train_ELAM_LSTM.py   # Main training script
├── requirements.txt     # Python dependencies
├── data/                # Data directory (must be created by user)
│   └── Tracks_Versuche_A_mit_CFD.csv # Fish tracking data
└── results/             # Output directory (automatically created)
