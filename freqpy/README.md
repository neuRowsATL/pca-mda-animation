
		FreqPy

===============================
		README
===============================


Contents
===============================
1. Description
2. Summary
3. Data Folder Setup


1. Description
===============================

FreqPy is a GUI for analyzing frequency dependence in networks of neurons. 
The goal of this project is to provide an environment for quick, 
qualitative analysis that might lead to further exploration.


2. Summary (Tabs)
===============================

[Analyze]:
The following analysis methods can be viewed as a 3D rotatable image:
- PCA
- MDA
- ICA
- K-Means (shows a comparison between k-means labels and original labels)

[Categorize]:
Displays a raster plot of all neurons in the dataset.

[Visualize]:
The results from the analysis methods above can be exported to video.

[Clusterize]:
Each neuron is marked as active or inactive depending its average change in frequency for each class.
	- e.g. Neuron 1 is active for class 0 because its average change in frequency is 0.2. 
	But Neuron 1 is inactive for class 1 because its average change in frequency in response to the stimulus
	is 0.01. The ability to change the threshold for active/inactive is coming...

[Compare-ize]:
Shows the similarity between each cluster (class) using a variety of similarity metrics.


3. Data Folder Setup
===============================

- All of your neural data files as '.txt' (on-times for each neuron) in this format:
	"YYYYMMDD_NAME-##.txt"
	E.g. "20120411_CBCO-1.txt"

- The input labels file in CSV format as a '.txt', which must be called:
	"pdat_labels.txt"
	
	OR

	The labels file can be created for you from a list of spike times.
	If you choose this option, the file must be formatted as a comma separated list,
	including the column labels in the following order:
	
		ON TIME,OFF TIME,LABEL
		12444,40000,1
		12222,12333,2
		50000,60000,2

	Save this file as a '.txt' or '.csv' and select it when prompted.
	A 'pdat_labels.txt' file will be created in your data folder.


- The waveform file called: 
		- "waveform.txt" (with N samples, where N is the number of samples in your dataset)
		OR
		- "*****.asc" (this can be exported directly from DataView)

- The name of each condition (class) should be in a file called "waveform_names.json". It should look like this:
	
	{"1": "no_sim", "2": "CL", "3": "low_sine", "4": "top_sine", "5": "inf_sine", "6": "tugs_ol", "7": "other"}
	
	
	*** IMPORTANT ***
	
	- Label "1" should correspond to "No Simulation". Otherwise it creates problems for the [Clusterize] tab, which
	assumes "1" is a baseline.

	- Labelling should always begin with 1 and NOT 0.