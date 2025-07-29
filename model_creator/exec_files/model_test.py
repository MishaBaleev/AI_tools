import time
import cv2
from ultralytics import YOLO
import torch
torch.set_num_threads(1)
from Logger import Logger

WINDOW_NAME = "YOLO test"
TARGET_FPS = 60
FRAME_TIME = 1/TARGET_FPS
RESOLUTION = (1280, 720)
IMGSZ = 320
CONF_THRESH = 0.5
LOGGER = Logger(logger_name="YOLO_test").logger

class FPS_monitor:
    def __init__(self, avg_window=10) -> None:
        self.times = []
        self.avg_window = avg_window
    
    def update(self) -> None:
        self.times.append(time.time())
        if len(self.times) > self.avg_window: self.times.pop(0)
    
    def get_fps(self) -> int:
        if len(self.times) < 2: return 0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])

def init_model(model_path:str) -> YOLO:
    model = YOLO(model_path)
    model.fuse()
    LOGGER.info(f"YOLO-model is loaded")
    return model

def process_frame(frame, model:YOLO):
    frame = cv2.resize(frame, RESOLUTION, interpolation=cv2.INTER_AREA)
    results = model.predict(
        frame,
        imgsz=IMGSZ,
        conf=CONF_THRESH,
        augment=False,
        verbose=False
    )
    for result in results:
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            bbox = [int(coord) for coord in box.xyxy[0].tolist()]
            LOGGER.warning(f"class - {class_id} class_name - {class_name} conf - {confidence:.2f} bbox - {bbox}")  
    return results[0].plot()

def get_video(video_path:str, model:YOLO):
    fps_monitor = FPS_monitor()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        LOGGER.critical("can't open video-file")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paused = False
    step_mode = False
    speed = 1.0
    manual_frame_change = False
    try:
        while True:
            if not paused or manual_frame_change:
                start_time = time.perf_counter()
                if manual_frame_change: manual_frame_change = False
                else:
                    ret, frame = cap.read()
                    if not ret: 
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                ret, frame = cap.retrieve() if manual_frame_change else (ret, frame)
                if not ret: continue
                annotated_frame = process_frame(frame, model)
                fps_monitor.update()
                current_fps = fps_monitor.get_fps()
                controls = [
                    "Space: Play/Pause",
                    "R: Rewind",
                    "Q: Quit"
                ]
                for i, control in enumerate(controls):
                    cv2.putText(annotated_frame, control, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.putText(annotated_frame, f"Frame: {current_frame}/{total_frames}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(WINDOW_NAME, annotated_frame)
                if not paused and not step_mode:
                    elapsed = time.perf_counter() - start_time
                    delay = max(1, int((FRAME_TIME/speed - elapsed) * 1000))
                    key = cv2.waitKey(delay)
                else: key = cv2.waitKey(0)
            else: key = cv2.waitKey(0)
            if key == ord(' '): paused = not paused
            elif key == ord('q'): break
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                manual_frame_change = True
    except KeyboardInterrupt: pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main(model_path:str) -> None:
    video_path = input("\nEnter path to video: ").replace("'", "").replace('"', '')
    model = init_model(model_path)
    get_video(video_path=video_path, model=model)
    
if __name__ == "__main__":
    art = '''
   _____ ___________   _____    __       ____  __________
  / ___//  _/ ____/ | / /   |  / /      / __ )/  _/_  __/
  \__ \ / // / __/  |/ / /| | / /      / __  |/ /  / /
 ___/ // // /_/ / /|  / ___ |/ /___   / /_/ // /  / /
/____/___/\____/_/ |_/_/  |_/_____/  /_____/___/ /_/
    '''
    print(art)
    model_path = input("\nEnter path to YOLO-model: ").replace("'", "").replace('"', '')
    main(model_path=model_path)