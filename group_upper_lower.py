import os
import numpy
import datetime

now = datetime.datetime.now()
nowDatetime = now.strftime('%m%d_%H%M%S')

def getFrames(ff):
    f = open(ff, 'r')
    for line in f:
        if "Frames:" in line:
            f.close()
            return int(line.split(' ')[1][:-1])

def hipMove(motion):
    motion = motion.split(' ')
    return ' '.join(motion[:6])

def otherMove(motion):
    motion = motion.split(' ')
    return ' '.join(motion[6:])[:-1]

def save_full_motion(lower_path, upper_path, output_path):
    files = os.listdir(lower_path)
    print("saving full motion")
    lower_path = lower_path + '/' + files[0]
    print(lower_path)
    print(upper_path)
    print(output_path)
    lower = open(lower_path,"r")
    upper = open(upper_path ,"r")
    out = open(output_path + "/" + nowDatetime +"_fullbody.bvh","w")
    
    lower_frame = getFrames(lower_path)
    upper_frame = getFrames(upper_path)

    lower_lines = lower.readlines()
    upper_lines = upper.readlines()

    l_idx = 0
    for line in lower_lines:
        l_idx += 1
        out.write(line)
        if l_idx==63:
            l_idx+=4
            break
    
    u_idx = 0
    joint = False

    for line in upper_lines:
        u_idx += 1
        if "LowerBack" in line:
            joint=True
        if not joint:
            continue
        out.write(line)
        if "Frame Time" in line:
            break
    
    for i in range(upper_frame):
        out.write(hipMove(lower_lines[l_idx + i]) + ' ' + otherMove(lower_lines[l_idx + i]) + ' ' + otherMove(upper_lines[u_idx + i]) + '\n')
    print("save finished")

    