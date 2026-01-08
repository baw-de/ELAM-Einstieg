# Fish Movement Prediction: ELAM & ELAM-LSTM

This repository contains two distinct approaches for predicting fish swimming behavior in response to complex flow fields, developed at the **Bundesanstalt für Wasserbau (BAW)**.

---

## Project Structure

The repository is organized into two primary modules representing Modeling upstream fish movement in 2D in a laboratory experiment using the Eulerian-Lagrangian-agent method (ELAM). 

### 1. ELAM (Rule-Based)
The **Eulerian-Lagrangian-agent method (ELAM) ** folder.
* **Methodology**: Implements a rule-based system where fish movement is simulated based on hydraulic stimuli.
* **Logic**: Decisions are driven by predefined behavioral rules and biological thresholds in response to environmental gradients.
See the PhD of David Gisen for a complete description of the setup and goals: 
Modeling upstream fish migration in small-scale using the Eulerian-Lagrangian-agent method (ELAM) [https://hdl.handle.net/20.500.11970/105158]

### 2. ELAM-LSTM (Deep Learning)
A modern approach using Recurrent Neural Networks to model movement sequences.
* **Methodology**: Utilizes a Long Short-Term Memory (LSTM) network to predict swimming velocity components ($u_{swim}$, $v_{swim}$).
* **Feature Integration**: Directly learns from Computational Fluid Dynamics (CFD) data, including velocity fields ($U, V$) and Total Kinetic Energy (TKE).

## Authors and acknowledgment
The code was implemented within a research and develoment project at the department of Hydraulic Engineering in Inland Areas at the german federal Waterways Engineering and Research Institute (https://www.baw.de)

## License
GNU General Public License 3
https://www.gnu.org/licenses/gpl-3.0.html



