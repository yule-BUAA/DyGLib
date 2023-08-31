import numpy as np
import pandas as pd
from tabulate import tabulate


def pprint_df(df, tablefmt='psql'):
    print(tabulate(df, headers='keys', tablefmt=tablefmt))


if __name__ == "__main__":
    all_datasets = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                    'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts']
    records = []
    for dataset_name in sorted(all_datasets, key=lambda v: v.upper()):
        edge_raw_features = np.load('../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
        info = {'dataset_name': dataset_name,
                'num_nodes': node_raw_features.shape[0] - 1,
                'node_feat_dim': node_raw_features.shape[-1],
                'num_edges': edge_raw_features.shape[0] - 1,
                'edge_feat_dim': edge_raw_features.shape[-1]}
        records.append(info)

    info_df = pd.DataFrame.from_records(records)
    pprint_df(info_df)
