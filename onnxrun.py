import os
import cv2
import numpy as np
import onnxruntime
import time


class Onnx():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath, providers=['CUDAExecutionProvider'])
        self.input_name = []
        for node in self.onnx_session.get_inputs():
            self.input_name.append(node.name)
            
        self.output_name = []
        for node in self.onnx_session.get_outputs():
            self.output_name.append(node.name)
        
    # -------------------------------------------------------
    #  预处理
    # -------------------------------------------------------
    def pro_(self,img):
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       img = img.transpose(2, 0, 1) 
       img = img.astype(dtype=np.float32)
       img /= 255.0
       img = np.expand_dims(img, axis=0)

       r1i = np.array([[[[0,]]]],np.float32)
       r2i = np.array([[[[0,]]]],np.float32)
       r3i = np.array([[[[0,]]]],np.float32)
       r4i = np.array([[[[0,]]]],np.float32)
       
       downsample_ratio = np.array([0.99,],np.float32)
      
       return img,r1i,r2i,r3i,r4i,downsample_ratio


    # -------------------------------------------------------
    #  推理
    # -------------------------------------------------------
    def __call__(self, img):
        src,r1i,r2i,r3i,r4i,downsample_ratio = self.pro_(img)
        input_feed = {"src":src,"r1i":r1i,"r2i":r2i,"r3i":r3i,"r4i":r4i,"downsample_ratio":downsample_ratio}
        pred = self.onnx_session.run(None, input_feed)
        return pred


if __name__ == "__main__":
    model = Onnx("./model.onnx")
    
    '''img = cv2.imread("./bus.jpg")
    fgr,pha,pha_sm,fgr_sm,err_sm,ref_sm = [i[0].transpose(1, 2, 0)  for i in model(img)]
    print([i.shape for i in [pha,fgr,pha_sm,fgr_sm,err_sm,ref_sm]])
    new_img = fgr[...,::-1]*pha
    cv2.imshow("pha",pha)
    cv2.imshow("new",new_img)
    cv2.waitKey(0)'''
    
    # 创建视频捕获对象
    #cap = cv2.VideoCapture('rtsp://172.16.25.207')
    cap = cv2.VideoCapture('/home/disk/wcp/Source/movie/Teacher2.mp4')
    # 检查视频捕获对象是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()
    
    while cap.isOpened():
        # 读取帧
        ret, frame = cap.read()
        # 检查是否成功读取帧
        if not ret:
            print("无法读取帧或视频播放完毕")
            break
        
        fgr,pha,r1o,r2o,r3o,r4o = [i[0].transpose(1, 2, 0)  for i in model(frame)]
        new_img = fgr[...,::-1]*pha
        
        h,w = new_img.shape[0],new_img.shape[1]
        pha = cv2.resize(new_img,(int(w/2),int(h/2)))
        # 在窗口中显示帧
        cv2.imshow('pha', pha) 
        cv2.imshow('frame', frame) 
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     

    # 释放捕获对象和关闭窗口
    #cap.release()
    cv2.destroyAllWindows()
