import os

for name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'myket',
             'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts']:
    os.system(f'python preprocess_data.py  --dataset_name {name}')
