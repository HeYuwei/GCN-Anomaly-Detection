import os

mpath = '/mmu_ssd/liuchang03/heyuwei/temporal-segment-networks/3rd-party/opencv-3.4.5/modules'
for root,dirs,files in os.walk(mpath):
    for dir in dirs:
        if dir == 'opencv2':
            s_path = os.path.join(root,dir)
            cmd = 'cp -r ' + s_path + ' /usr/local/include'
            # cmd = 'cp -r ' + s_path + ' /mmu_ssd/liuchang03/heyuwei/anaconda3/envs/hyw27/include'
            print(cmd)
            os.system(cmd)