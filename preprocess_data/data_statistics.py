import numpy as np

if __name__ == "__main__":
    for dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                         'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts']:
        edge_raw_features = np.load('../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

        print('Statistics of dataset ', dataset_name)
        print('number of nodes ', node_raw_features.shape[0] - 1)
        print('number of node features ', node_raw_features.shape[1])
        print('number of edges ', edge_raw_features.shape[0] - 1)
        print('number of edge features ', edge_raw_features.shape[1])
        print('====================================')
