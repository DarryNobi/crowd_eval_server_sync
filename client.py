import cv2
import json
import requests
import sys

def main():
    if(len(sys.argv)<2):
        print("使用方法：python client.py path2image.jpg")
        return 0
    path=sys.argv[1]
    img = cv2.imread(path)
    img=cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
    res = {'image':str(img.tolist())}
    #print(res)
    data = json.dumps(res)
    # cls = requests.post("http://139.198.18.130:8082", data=json.dumps(res))
    cls = requests.post("http://172.20.53.158:8082", data=json.dumps(res))
    print(["状态码：",cls.status_code])
    print(["分类结果:",cls.text,"级拥挤"])

    cv2.imshow("img",img)
    cv2.waitKey(0)

if(__name__=='__main__'):
    main()
