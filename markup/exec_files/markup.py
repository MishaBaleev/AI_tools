import cv2
import os
import ujson as json
import numpy as np
import time

def valid_get_config() -> list or None:
    try: 
        with open("./config.json", "r") as config:
            classes = json.loads(config.read())["classes"]
            return classes
    except ValueError as e:
        print(f"Error with config - {e}")
        return None

class VideoAnnotator:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(video_path)
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.target_width = 1280
        self.target_height = 720
        self.current_frame = 0
        time_stamp = str(time.time())
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        self.images_dir = os.path.join(self.images_dir, time_stamp)
        self.labels_dir = os.path.join(self.labels_dir, time_stamp)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        self.classes = valid_get_config()
        self.current_class = 0
        self.annotations = {}
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.temp_boxes = []
        self.final_boxes = []
        self.scale_x = self.target_width / self.orig_width
        self.scale_y = self.target_height / self.orig_height
    
    def resize_frame(self, frame:np.array) -> np.array:
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_boxes.append({'start': (x, y), 'end': (x, y), 'class_id': self.current_class})
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing: self.temp_boxes[-1]['end'] = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.temp_boxes[-1]['end'] = (x, y)
            x1, y1 = self.temp_boxes[-1]['start']
            x2, y2 = self.temp_boxes[-1]['end']
            self.temp_boxes[-1]['start'] = (min(x1, x2), min(y1, y2))
            self.temp_boxes[-1]['end'] = (max(x1, x2), max(y1, y2))
    
    def draw_boxes(self, frame:np.array) -> None:
        for box in self.temp_boxes:
            x1, y1 = box['start']
            x2, y2 = box['end']
            class_id = box['class_id']
            class_name = self.classes[class_id] if class_id < len(self.classes) else str(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        for box in self.final_boxes:
            x1, y1 = box['start']
            x2, y2 = box['end']
            class_id = box['class_id']
            class_name = self.classes[class_id] if class_id < len(self.classes) else str(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    def draw_controls(self, frame:np.array) -> None:
        controls = [
            f"{', '.join([f'{index} - {item}' for index, item in enumerate(self.classes)])}",
            "Controls:",
            "LMB - draw bounding box",
            "s - save current bboxes",
            "n/p - next/previous frame",
            "d - delete last bbox",
            "c - clear all bboxes",
            f"0-{len(self.classes)-1} - select class",
            "q - quit"
        ]
        overlay_1 = frame.copy()
        cv2.rectangle(overlay_1, (10, 10), (400, 40 + len(controls)*25), (0, 0, 0), -1)
        cv2.addWeighted(overlay_1, 0.7, frame, 0.3, 0, frame)
        overlay_2 = frame.copy()
        cv2.rectangle(overlay_2, (401, 10), (800, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay_2, 0.7, frame, 0.3, 0, frame)
        for i, text in enumerate(controls):
            y = 40 + i * 25
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        class_text = f"Current class: {self.classes[self.current_class]} ({self.current_class})"
        cv2.putText(frame, class_text, (20, 40 + len(controls)*25 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        frame_text = f"Frame: {self.current_frame}/{self.total_frames}"
        cv2.putText(frame, frame_text, (self.target_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def convert_to_yolo_format(self, box:dict, img_width:int, img_height:int) -> str:
        x1, y1 = box['start']
        x2, y2 = box['end']
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return f"{box['class_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    
    def save_annotation(self) -> None:
        if not self.final_boxes: return
        ret, frame = self.cap.read()
        if not ret: return
        frame = self.resize_frame(frame)
        frame_filename = f"frame_{self.current_frame:06d}.jpg"
        frame_path = os.path.join(self.images_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        label_filename = f"frame_{self.current_frame:06d}.txt"
        label_path = os.path.join(self.labels_dir, label_filename)
        height, width = frame.shape[:2]
        with open(label_path, 'w') as f:
            for box in self.final_boxes:
                yolo_line = self.convert_to_yolo_format(box, width, height)
                f.write(yolo_line + "\n")
        self.annotations[self.current_frame] = {
            'image_path': frame_path,
            'label_path': label_path,
            'boxes': self.final_boxes.copy()
        }
    
    def run(self) -> None:
        cv2.namedWindow("Video Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Video Annotator", self.mouse_callback)
        self.current_frame = (self.current_frame // 10) * 10
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read video")
            return
        frame = self.resize_frame(frame)
        display_frame = np.zeros_like(frame)
        while True:
            display_frame[:] = frame.copy()
            self.draw_boxes(display_frame)
            self.draw_controls(display_frame)
            cv2.putText(display_frame, "10x FRAME MODE", (20, self.target_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Video Annotator", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('n'):
                self.save_annotation()
                self.current_frame = min(self.current_frame + 10, self.total_frames - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if ret:
                    frame = self.resize_frame(frame)
                self.temp_boxes = []
                self.final_boxes = []
            elif key == ord('p'):
                self.save_annotation()
                self.current_frame = max(self.current_frame - 10, 0)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if ret:
                    frame = self.resize_frame(frame)
                self.temp_boxes = []
                self.final_boxes = []
            elif key == ord('s'):
                self.final_boxes.extend(self.temp_boxes)
                self.temp_boxes = []
            elif key == ord('d'):
                if self.temp_boxes: self.temp_boxes.pop()
                elif self.final_boxes: self.final_boxes.pop()
            elif key == ord('c'):
                self.temp_boxes = []
                self.final_boxes = []
            elif ord('0') <= key <= ord('9'):
                class_num = key - ord('0')
                if class_num < len(self.classes): self.current_class = class_num
        
        self.save_annotation()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    art = '''
   _____ ___________   _____    __       ____  __________
  / ___//  _/ ____/ | / /   |  / /      / __ )/  _/_  __/
  \__ \ / // / __/  |/ / /| | / /      / __  |/ /  / /
 ___/ // // /_/ / /|  / ___ |/ /___   / /_/ // /  / /
/____/___/\____/_/ |_/_/  |_/_____/  /_____/___/ /_/
    '''
    print(art)
    video_path = input("Enter path to video file: ").replace("'", "").replace('"', '')
    output_dir = "./raw_dataset"
    
    if not os.path.exists(video_path):
        print(f"Error: video file {video_path} not found!")
        exit()
    
    annotator = VideoAnnotator(video_path, output_dir)
    if annotator.classes != None:
        annotator.run()