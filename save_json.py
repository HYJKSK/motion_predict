
import json
import numpy as np
def save_data(X, y, path):
    #'data/dataset.json'
    data ={}
    X =  [_.tolist() for _ in X]
    y =  [_.tolist() for _ in y]
    data['X'] = X
    data['y'] = y
    json.dumps(data, ensure_ascii=False, indent="\t")
    with open(path, 'w', encoding="utf-8") as make_file:
        json.dump(data, make_file, ensure_ascii=False, indent='\t')
    print("save json")

def read_data(path):
    # 'data/dataset.json'
    print("read json data")
    with open(path) as file_json:
        data_json = json.load(file_json)
    return np.array(data_json['X']), np.array(data_json['y']) #X, y
