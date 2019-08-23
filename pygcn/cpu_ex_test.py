import cv2
import os
from multiprocessing import Pool, current_process

vid_path = '/mmu_ssd/liuchang03/heyuwei/svideos/Abuse001_x264.mp4'
out_full_path = '/mmu_ssd/liuchang03/heyuwei/dense_flow/tmp_data'


# current = current_process()
# dev_id = int(current._identity[0]) - 1
dev_id = 0
image_path = '{}/img'.format(out_full_path)
flow_x_path = '{}/flow_x'.format(out_full_path)
flow_y_path = '{}/flow_y'.format(out_full_path)

cmd = '/mmu_ssd/liuchang03/heyuwei/dense_flow/extract_cpu -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o=zip'.format(vid_path, flow_x_path, flow_y_path, image_path, dev_id)

os.system(cmd)