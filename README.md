# ELAM-Einstieg

Modeling upstream fish migration in 2D in a laboratory experiment using the Eulerian-Lagrangian-agent method (ELAM). 

#### Example: tracks of simulated agents

![Image](https://github.com/user-attachments/assets/f80469ec-0a69-4e16-9af8-d9f707ec06f2)


## Description
A python code to simulate the position and movement of an agent fish in a 2D grid based on flow field data, fish behavior models, and external conditions.
    
The simulation of the movement is in response to the surrounding flow field, incorporating three behavioral factors like migrating, holding, and drifting behaviors. 
The code also applies movement corrections, handles fish interactions with geometry and updates their locations within the defined polygon area.

See the PhD of David Gisen for a complete description of the setup and goals: 
Modeling upstream fish migration in small-scale using the Eulerian-Lagrangian-agent method (ELAM) [https://hdl.handle.net/20.500.11970/105158]

And the ELAM-de repository for a similar code in C++ and fortran:
https://github.com/baw-de/ELAM-flume

#### ELAM in a nutshell:

![Image](https://github.com/user-attachments/assets/d5a90958-23b3-4030-8aff-9fe168236a82)

## Example of output

#### Vector relation 

![Image](https://github.com/user-attachments/assets/d6112eec-fce0-42f4-8b0b-e9a14387567c)

#### Track of a single agent

![Image](https://github.com/user-attachments/assets/05461df4-6977-4dc2-bd82-da5a3ba25b36)


## Using the code
Define the path to the data and output path and **run the script _1_run_ELAM_flume_2D.py** to get the results.

#### Core functions
The script _0_function_ELAM_flume_2D.py contains the main functions needed for calculating the agent movement vectors and interaction with the geometry. Additional functions for plotting the results are included. The user is advised not to change the core functions. 

#### How to run the code?

The script _1_run_ELAM_flume_2D.py expects two files as input:

- The 2D flow field as .csv file with the following format:

| Index | xcfd_grid | ycfd_grid | u_f | v_f | U_mag | flow_angle
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0 | -4.5775 | 0.12243 | 0.40833 | 0.000111 | 0.408330 | 0.408330 |
| 1 | -4.5289 |   0.12177 | 0.40802 | 0.000401 | 0.408020  |  0.056362|


- A Shapefile of the geometry with the extension .shp

The input files should be located in the folder data (see folder 'data')

#### Parameters

Check in the script lines 93 to 146. These contain the parameters that can be changed by the user. A full description of the parameters can be found in https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263964

## Authors and acknowledgment
The code was implemented within a research and develoment project at the department of Hydraulic Engineering in Inland Areas at the german federal Waterways Engineering and Research Institute (https://www.baw.de)

## License
GNU General Public License 3




