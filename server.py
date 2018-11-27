#coding:utf-8
from flask import request,Flask
import json
import numpy as np
import crowd_eval
import cv2
from skimage import transform,data

app=Flask(__name__)

@app.route('/',methods=['POST'])
def get_frame():
    res = json.loads(request.data)
    #print("get json data:")
    #print(res)
    frame=eval(res["image"])
    frame=np.array(frame,dtype=np.uint8)
    frame=cv2.resize(frame,(299,299),interpolation=cv2.INTER_NEAREST)
    print(frame.shape,frame.dtype)
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hist,_=np.histogram(frame_gray.ravel(),32,[0,256])
    max_hist=np.max(hist)
    if(max_hist>60000):
        return '-1'
    res=crowd_eval.predict(frame)
    return str(res)

if __name__=='__main__':
    crowd_eval.load_model()
    app.run("172.16.10.2",port=8082)
