import datetime
import os

now = datetime.datetime.now()
nowDatetime = now.strftime('_%m%d_%H%M%S')
# file_dir = "dataset/mocap_xia/"
open_dir = "dataset/mocap_xia/"
dist_dir = "dataset/leg/"
file_name = "angry_01_000.bvh"
dst_name = file_name.split('.')[0] + nowDatetime + ".bvh"
left_joint_num = 11

def cutMotion(line, n):
    frame = line.split(' ')
    framecut = ' '.join(frame[:6 + 3*(n-1)]) + '\n'
    return framecut

def cutFile(f):
    src = open(open_dir + f, 'r')
    dst = open(dist_dir + f.split('.')[0] + "_leg.bvh", 'w')
    line_num = 0
    skip = False
    framepart = False
    for line in src:
        line_num += 1
        if not framepart:
            if "LowerBack" in line:
                skip = True
            if "MOTION" in line:
                dst.write('}\n')
                skip = False
            if "Frame Time" in line: # Motion 부분
                framepart = True
            if not skip:
                dst.write(line)
        else:
            cut = cutMotion(line, left_joint_num)
            dst.write(cut)
    src.close()
    dst.close()
        
def main():
    files = os.listdir(open_dir)
    n_files = len(files)
    i = 0
    for f in files:
        i += 1
        cutFile(f)
        if int(i/n_files*100)%10 == 0:
            print('----- {}% done'.format(int(i/n_files*100)))
        
main()
        