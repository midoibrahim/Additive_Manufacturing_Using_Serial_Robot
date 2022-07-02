import cv2
import torch
import requests
import winsound
import numpy as np
from time import time

class spaghetti_detector:
    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        self.model = torch.hub.load('./torch/hub/ultralytics_yolov5_master', 'custom', path=model_name, source='local')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.flag = 0
        self.counting_array = [0]
        self.weburl = 'http://192.168.130.126:5000/'

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, results, frame):
        flag = 0
        labels, cord = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(len(labels)):
            row = cord[i]
            if row[4] >= 0.7:
                flag = 1
                display_text = self.model.names[int(labels[i])]
                ((label_width, label_height), _) = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_PLAIN, 1.6, 2)
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (int(x1 + label_width * 1.05), int(y1 + label_height * 1.25)), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, display_text, (x1, int(y1 + label_height * 1.2)), cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 0), 2)
        self.counting_array.append(flag)
        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        while True:
            ret, frame = cap.read()
            assert ret
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.flag > 0:
                cv2.putText(frame, 'Spaghetti Detected!', (10 , 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
                self.flag = self.flag - 1
            cv2.imshow('YOLOv5 Detection', frame)

            if self.counting_array[-1] == 0:
                self.counting_array.clear()
                self.counting_array.append(0)
            
            if len(self.counting_array) > 30:
                winsound.Beep(2500, 1000)
                self.flag = 20
                try:
                    get = requests.get(self.weburl)
                    if get.status_code == 200:
                        requests.get(self.weburl + 'pausespa')
                    else:
                        print(f"{self.weburl}: is Not reachable, status_code: {get.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"{self.weburl}: is Not reachable \nErr: {e}")
                self.counting_array.clear()
                self.counting_array.append(0)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()

detector = spaghetti_detector(capture_index = 0, model_name = 'weights.pt')
detector()