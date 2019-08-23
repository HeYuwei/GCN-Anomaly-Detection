import pickle
import os

train_pkl = 'ucf_crime_train.pkl'
root_video_folder = "/mmu_ssd/liuchang03/heyuwei/Data/crime_pic/"


with open(train_pkl, 'rb') as f:
    train_videos = pickle.load(f)

my_train_vpaths = []
my_valid_vpaths = []

for root, dirs, files in os.walk(root_video_folder):
    for d in dirs:
        if d.endswith('x264'):
            video_folder = os.path.join(root, d)
            video_folder = video_folder.replace(root_video_folder, '')

            if d in train_videos:
                print(d)
                my_train_vpaths.append(video_folder)
            else:
                my_valid_vpaths.append(video_folder)

print('train num ' + str(len(my_train_vpaths)))
print('valid num ' + str(len(my_valid_vpaths)))

with open('my_crime_train.pkl','wb') as f:
    pickle.dump(my_train_vpaths,f)

with open('my_crime_valid.pkl','wb') as f:
    pickle.dump(my_valid_vpaths,f)