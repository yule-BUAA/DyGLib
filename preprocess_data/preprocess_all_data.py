import os

for name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo',
             'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts']:
    os.system(f'python preprocess_data.py  --dataset_name {name}')
