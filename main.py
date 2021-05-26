import numpy as np
import os
import argparse
import importlib
from os.path import join as pjoin
# from remove_fs import remove_fs, save_bvh_from_network_output


from footContact import getFootPosition
from group_upper_lower import save_full_motion

from data_loader import process_single_bvh, save_bvh_from_network_output, content_and_phase

from save_json import save_data, read_data

# model 관련 라이브러리
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from model_utils import get_history

from tensorflow import keras

# 현재 시간 저장
import datetime
now = datetime.datetime.now()
nowDatetime = now.strftime('%m%d_%H%M%S')

# 전역 변수
frames_over = 200
output_fnum = 200
home_path = "C:/Users/Seogki/GoogleDrive/HY/graduationProject/motion_predict/"
home_path = "" 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='foot_print')

    return parser.parse_args()


def getFrame(path, filename):
    f = open(path+"/"+filename, 'r')
    for line in f:
        if "Frames:" in line:
            f.close()
            return int(line.split(' ')[1][:-1])

def get_dataset(path, fnum, bodytype, is_y = False):
    print("get dataset is_y ==",is_y)
    files = os.listdir(path)
    files.sort()
    dataset=[]
    for f in files:
        if f == '.DS_Store':
            continue
        if getFrame(path, f) < frames_over:
            continue
        if is_y == True : # y일 때
            data = process_single_bvh(path+"/"+f, bodytype=bodytype, to_batch=True)
            data = data['contentraw'].numpy().reshape(data['contentraw'].shape[1],data['contentraw'].shape[2])[:,:fnum]
            data = data.flatten()
        else: # X 일때
            if args.type == 'upper_body':
                data = content_and_phase(path+"/"+f, bodytype=bodytype)[:,:fnum]
            else :
                l, r, foot_step = getFootPosition(path+"/"+f)
                l = l[:fnum]
                r = r[:fnum]
                data = np.concatenate((l,r),axis=1).reshape(fnum,8).T
            # (#Data, 2, fnum*4)
        dataset.append(data)
    return np.array(dataset)

def get_cnn_model(n_inputs, n_outputs):
    model = Sequential([
        layers.Conv1D(256, 3, activation='relu', input_shape = n_inputs),
        layers.Conv1D(128, 2, activation='relu'),
        layers.MaxPooling1D(pool_size=2, padding = 'same'),
        layers.Conv1D(32, 3, activation='relu'),
        layers.Dropout(0.5),
        layers.MaxPooling1D(pool_size=2, padding = 'same'),
        layers.Flatten(),
        Dense(100, activation = 'relu'),
        Dense(n_outputs)
    ])
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def save_model(X_train, y_train, X_test, y_test, n_inputs, n_outputs, fnum, name):
    # get model
    model = get_cnn_model(n_inputs, n_outputs)

    # fit the model on all data
    epochs = 2000

    history = model.fit(X_train, y_train, validation_data =(X_test, y_test),verbose=1, epochs=epochs)
    get_history(history, epochs)

    #save model
    model.save(name)
    print("model name :  ", name)

    return model

def load_model(name):
    model = keras.models.load_model(name)
    return model

def fit_model(model, fnum, path, bodytype=0):
    # make a prediction for new data
    row = get_dataset(path, fnum, True, is_y=False) # "dataset/predict"

    yhat = model.predict(row)

    # save output motion
    save_motion("output_"+nowDatetime+".bvh",yhat[0], fnum, bodytype=bodytype, leg_path=path)
    print("save predict motion")

def save_motion(name, motion, fnum, bodytype=0, leg_path=""):
    joint = None
    if bodytype==1:
        joint = 48
    elif bodytype==2:
        joint = 88
    else:
        joint = 128
    
    motion = motion.reshape(joint,fnum) # 128, 48
    output_dir = home_path + "output"
    save_bvh_from_network_output(motion, output_path=pjoin(output_dir, name), bodytype=bodytype)
    if args.type == 'upper_body':
        save_full_body_path = home_path + "output/full_body"
        save_full_motion(leg_path, pjoin(output_dir, name) , save_full_body_path)


def main(args):

    json_path = home_path +"dataset/json/" + str(frames_over) + '_f' + str(output_fnum) + "XfootContactByOriginYBodyZXY.json" #seogki 
    
    if args.type == 'upper_body':
        X_path = "dataset/leg"
        y_path = "dataset/upper_body"
        bodytype = 2
    else : 
        X_path = "dataset/leg"
        y_path = "dataset/raw"
        bodytype = 0

    fnum = output_fnum

    # generate new data
    if(args.type == 'upper_body'):
        X = get_dataset(X_path, fnum, 1)
        y = get_dataset(y_path, fnum, 2, is_y=True)
    else: # foot_print
        X = get_dataset(X_path, fnum, -1)
        y = get_dataset(y_path, fnum, 0, is_y=True)

    # save json dataset
    # save_data(X, y, json_path) 

    # use json dataset
    # X, y = read_data(json_path)

    print("X.shape, y.shape " , X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)
    
    print("X_train, X_test, y_train, y_test ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    n_inputs, n_outputs = X_train.shape[1:], y_train.shape[1] # 2D for CNN

    print("n_inputs, n_outpouts", n_inputs, n_outputs)
    model_name = home_path+'model/'+nowDatetime+'_cnn_f'+ str(output_fnum)+'.h5' # 모델 이름
    # load_model_name = home_path+'model/0411_001945_cnn_f200.h5'
    model = save_model(X_train, y_train, X_test, y_test, n_inputs, n_outputs, fnum, model_name)
    # model = load_model(load_model_name)

    test_path = home_path+"dataset/predict"
    fit_model(model, fnum, test_path, bodytype=bodytype)

if __name__ == '__main__':
    args = parse_args()
    main(args)



