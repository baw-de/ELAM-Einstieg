# ELAM-LSTM: Fish Movement Prediction

This repository contains the implementation of **ELAM-LSTM**, a deep learning model designed to predict fish swimming components ($u_{swim}$, $v_{swim}$) based on flow field features and historical movement data.

## Overview
### TRAIN_ELAM_LSTM.py

The model uses a Long Short-Term Memory (LSTM) network to process sequential data from fish tracks. It includes:
* **Custom LSTM Architecture:** Manages persistent hidden/cell states for continuous track prediction.
* **Physics-Informed Data:** Integrates Computational Fluid Dynamics (CFD) flow field data.
* **Robust Training:** Implements early stopping, learning rate scheduling, and dropout for regularization.

### Apply_ELAM_LSTM.py
The trained model is applied to generate agent tracks:
* **The Environment (CFD):** The simulation utilizes **DDES (Delayed Detached Eddy Simulation)** flow fields. The script reads these fields (stored as `.csv` or `.parquet`) and builds a **KDTree** for every time step. This allows the agents to "sense" the hydraulic conditions at their exact spatial coordinates efficiently.

* **The Agent (LSTM)**: Instead of hard-coded rules, the fish's "decision" is a pre-trained **LSTM model** (`.pt` file). 
- **Inputs:** Local $U, V, U_{mag}$, $k$ (turbulence), and the agent's previous swimming vector.
- **Output:** A predicted "swim vector" ($u_{swim}, v_{swim}$).

* **. Integration**: The position of each agent is updated using the formula:
$$xy_{t+1} = xy_t + (V_{swim} + V_{flow}) \times \Delta t$$
Where $\Delta t$ is matched to the CFD time step (approx. $1/19$ seconds).

---
## Project Structure

```text
├── Train_ELAM_LSTM.py   # Main training script
├── requirements.txt     # Python dependencies
├── data/                # Data directory (must be created by user)
│   └── Tracks_Versuche_A_mit_CFD.csv # Fish tracking data
└── results/             # Output directory (automatically created)
