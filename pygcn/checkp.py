import pickle

videos_pkl = 'ucf_crime_train.pkl'

with open(videos_pkl, 'rb') as f:
    videos = pickle.load(f)
for v in videos:
    print(v)