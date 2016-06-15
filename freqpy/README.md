										===============================
												* --- FreqPy --- *
										===============================

										===============================
												* --- README --- *
										===============================

===============================
General description
===============================

FreqPy is a GUI for analyzing frequency dependence in subpopulations of neurons. The goal of this project is to provide an environment for quick, qualitative analysis that might lead to further exploration.


===============================
Summary (Tabs)
===============================

[Analyze]:
	The following analysis methods can be viewed as a 3D rotatable image:

		- PCA
		- MDA
		- ICA
		- GMM
		- K-Means (PCA) (a comparison between k-means-produced classes and original input classes)

[Categorize]:
	Displays a raster plot of all neurons in the dataset. More features coming...

[Visualize]:
	The results from the analysis methods above can be exported to video.

[Clusterize]:
	Each neuron is marked as active or inactive depending its average change in frequency for each class.
		- e.g. Neuron 1 is active for class 0 because its average change in frequency is 0.2. 
		But Neuron 1 is inactive for class 1 because its average change in frequency in response to the stimulus
		is 0.01. The ability to change the threshold for active/inactive is coming...

[Compare-ize]:
	Shows the similarity between each cluster (class) using the Davies Bouldin Index (DBI)


===============================
Data Folder Setup
===============================

You'll need the following files in the data folder:

	- All of your neural data files as '.txt' (on-times for each neuron) in this format:
		"YYYYMMDD_NAME-##.txt"
		E.g. "20120411_CBCO-1.txt"
	
	- The input labels file in CSV format (N x m) as a '.txt', which must be called:
		"pdat_labels.txt"

	- The waveform file that contains N samples of the waveform, where N is the number of time points. The waveform data must be the same length as the input labels data array (e.g. if you have 1000 labels, your waveform file should contain 1000 samples of the original waveform data). The file must be named:
		"waveform.txt"

	- The name of each condition (class) should be in "waveform_names.json". It should look like this:
		
		{"1": "no_sim", "2": "CL", "3": "low_sine", "4": "top_sine", "5": "inf_sine", "6": "tugs_ol", "7": "other"}
		
		
		*** IMPORTANT ***
		
		- Label "1" should correspond to "No Simulation". Otherwise it creates problems for the [Clusterize] tab, which
		assumes "1" is a baseline.

		- Labelling should always begin with 1.
		
		*****************