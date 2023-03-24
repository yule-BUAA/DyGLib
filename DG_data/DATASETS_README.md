## Descriptions of data files in the paper *Towards Better Evaluation for Dynamic Link Prediction*

The original networks are saved as <network>.csv.

The networks are formatted as follows:
	* Each edge is denoted in one line.
	* Each line has the following format: source_node, destination_node, timestamp, edge_label, comma-separated arrays of edge features.
		* Please note that if there is no edge label available, the edge_label column will be filled with 0s only for loading purpose; these labels are not used in the link prediction task.
	* The first line denotes the network format.
	* Edge features should include at least one feature. If there is no edge feature available, a 0 value is used for all the edges.

The network edge-lists can be pre-processed by ```preprocess_data/preprocess_data.py```.
After pre-processing the network edge-list, there are three files that are used by the models:
	* <ml_network>.csv: this file contains the timestamp edge-list.
	* <ml_network>.npy: this file contains the edge features in the dense `npy` format that has the features in binary format.
	* <ml_network_node>.npy: this file contains the node features in the dense `npy` format that contains the node features in binary format.
Please note that when the edge features or node features are absent, we use a vector of zeros is used as the node/edge features in line with the baseline methods.
