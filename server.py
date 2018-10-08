#coding:utf-8
from flask import request,Flask
import json
import numpy as np
import crowd_eval

app=Flask(__name__)

@app.route('/',methods=['POST'])
def get_frame():
    res = json.loads(request.data)
    frame=eval(res["image"])
    frame=np.array(frame,dtype=np.uint8)
    res=crowd_eval.predict(frame)
    return str(res)

if __name__=='__main__':
    app.run("172.16.10.2",port=8082)
