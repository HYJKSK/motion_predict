import datetime
import os

now = datetime.datetime.now()
nowDatetime = now.strftime('_%m%d_%H%M%S')
# file_dir = "dataset/mocap_xia/"
predict_dir = "C:/Users/Seogki/GoogleDrive/HY/graduationProject/predict/"
open_dir = "dataset/mocap_xia/"
dist_dir = "dataset/upperbody/"
left_joint_num = 11

def cutMotion(line, n):
    frame = line.split(' ')
    framecut = ' '.join(frame[:6]) + ' ' + ' '.join(frame[6 + 3*(n-1):])
    return framecut

def cutFile(f):
    #src = open(open_dir + f, 'r')
    #dst = open(dist_dir + f.split('.')[0] + "_top.bvh", 'w')
    src = open(predict_dir + f, 'r')
    dst = open(predict_dir + f.split('.')[0] + "_top.bvh", 'w')
    line_num = 0
    skip = False
    framepart = False
    for line in src:
        line_num += 1
        if not framepart:
            if "LHipJoint" in line:
                skip = True
            if "LowerBack" in line:
                skip = False
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
    print("finished ", f)
    src.close()
    dst.close()
        
def main():
    files = os.listdir(predict_dir + open_dir)
    n_files = len(files)
    cutFile("rest.bvh")
    #for f in files:
    #    cutFile(f)
    print("-- cutting lower body finished --")
        
main()
        