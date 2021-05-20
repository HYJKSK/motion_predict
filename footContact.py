import numpy as np
from Quaternions_old import Quaternions
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

seogki_path ="C:/Users/Seogki/GoogleDrive/HY/graduationProject/predict/"
file_dir = "dataset/"
file_name = "angry_01_001_leg.bvh"

def translationMatrix(pos):
    x = float(pos[0])
    y = float(pos[1])
    z = float(pos[2])
    T = np.matrix([[ 1.0, 0.0, 0.0, x],
                   [ 0.0, 1.0, 0.0, y],
                   [ 0.0, 0.0, 1.0, z],
                   [ 0.0, 0.0, 0.0, 1.0]])
    return T

def rotationMatrixByQuaternion(rotation, order ='xzy', world = False):
    return np.array(Quaternions.from_euler(np.radians(rotation), order=order, world=world))

def getMotionFile(motion):
    f = open(motion,"r")
    
    boneLength = []
    pos = []
    rot = []
    mocap = False

    for line in f:
        if not mocap:
            if "OFFSET" in line:
                boneLength.append(line.split(' ')[1:])
            if "Frames" in line:
                frames = int(line.split(' ')[-1])
            if "Frame Time" in line:
                mocap = True
                continue
        else:
            frame = line.split(' ')
            rootPos = [frame[0],frame[1],frame[2],1]
            pos.append(rootPos)
            rot.append(frame[3:])
            
    return np.array(boneLength, dtype=np.float64), np.array(pos, dtype=np.float64), np.array(rot,dtype=np.float64), frames


def rotationMatrix(angle):
    angle = np.radians(angle)
    z = float(angle[0])
    y = float(angle[1])
    x = float(angle[2])
    X = np.matrix([[ 1.0, 0.0           , 0.0           ,0.0],
                   [ 0.0, np.cos(x)    ,-np.sin(x),0.0],
                   [ 0.0, np.sin(x)    , np.cos(x),0.0],
                   [0.0,0.0,0.0,1.0]])
    Y = np.matrix([[ np.cos(y), 0.0, np.sin(y),0.0],
                   [ 0.0        , 1.0, 0.0     ,   0.0],
                   [-np.sin(y), 0.0, np.cos(y),0.0],
                   [0.0,0.0,0.0,1.0]])
    Z = np.matrix([[ np.cos(z), -np.sin(z), 0.0,0.0 ],
                   [ np.sin(z), np.cos(z) , 0.0,0.0 ],
                   [ 0.0           , 0.0            , 1.0,0.0 ],
                   [0.0,0.0,0.0,1.0]])
    return np.dot(X, np.dot(Y, Z))

'''
[root, Lhip, LULeg, LLeg, LFoot, LToe, RHip, RULeg, RLeg, RFoot, RToe] # 11개
rotation
'''
def getFootPositionByOrigin(motion):
    translation, rootPosition, rotation, frames = getMotionFile(motion)

    leftFoot = []
    rightFoot = [] 
    footstep = []
    
    LHip_T = translationMatrix(translation[1])
    LULeg_T = translationMatrix(translation[2])
    LLeg_T = translationMatrix(translation[3])
    LFoot_T = translationMatrix(translation[4])
    LToe_T = translationMatrix(translation[5])
    RHip_T = translationMatrix(translation[6])
    RULeg_T = translationMatrix(translation[7])
    RLeg_T = translationMatrix(translation[8])
    RFoot_T = translationMatrix(translation[9])
    RToe_T = translationMatrix(translation[10])
    
    for frame in range(frames):
        root_P = np.matrix(rootPosition[frame], dtype=np.float64)
        root_R = rotationMatrix(rotation[frame,:3][::-1])
        LHip_R = rotationMatrix(rotation[frame,3:6][::-1])
        LULeg_R = rotationMatrix(rotation[frame,6:9][::-1])
        LLeg_R = rotationMatrix(rotation[frame,9:12][::-1])
        LFoot_R = rotationMatrix(rotation[frame,12:15][::-1])
        LToe_R = rotationMatrix(rotation[frame,15:18][::-1])
        RHip_R = rotationMatrix(rotation[frame,18:21][::-1])
        RULeg_R = rotationMatrix(rotation[frame,21:24][::-1])
        RLeg_R = rotationMatrix(rotation[frame,24:27][::-1])
        RFoot_R = rotationMatrix(rotation[frame,27:30][::-1])
        RToe_R = rotationMatrix(rotation[frame,30:][::-1])
        
        leftTrans = LToe_T@LToe_R@LFoot_R@LFoot_T@LLeg_R@LLeg_T@LULeg_R@LULeg_T@LHip_R@LHip_T@root_R
        leftHeel = leftTrans@root_P.T
        rightTrans = RToe_T@RToe_R@RFoot_R@RFoot_T@RLeg_R@RLeg_T@RULeg_R@RULeg_T@RHip_R@RHip_T@root_R
        rightHeel = rightTrans@root_P.T

        leftFoot.append(leftHeel)
        rightFoot.append(rightHeel)
        
    return np.array(leftFoot), np.array(rightFoot), np.array(footstep)

def getFootPosition(motion):
    return getFootPositionByOrigin(motion)


def drawHeight(l, r):
    fff = list(range(1,272))
    plt.plot(fff,np.array(r)[:,1,0])
    plt.title('both foot position per frame')
    plt.plot(fff,np.array(l)[:,1,0])
    plt.xlabel('frames')
    plt.ylabel('y value of foot position')
    plt.show()

def drawFootStep(l, r):
    print(np.array(r).shape, np.array(l).shape)
    plt.plot(np.array(r)[:,0,0],np.array(r)[:,2,0])
    plt.title('x, y position per frame')
    plt.plot(np.array(l)[:,0,0],np.array(l)[:,2,0])
    plt.xlabel('frames')
    #plt.ylabel('z value of foot position')
    plt.show()

def main():
    l,r,f= getFootPosition(file_dir + file_name)
    print(np.concatenate((l,r),axis=1).shape)
    drawHeight(l,r)

if __name__ == '__main__':
    main()