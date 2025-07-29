if __name__ == "__main__":
    art = '''
   _____ ___________   _____    __       ____  __________
  / ___//  _/ ____/ | / /   |  / /      / __ )/  _/_  __/
  \__ \ / // / __/  |/ / /| | / /      / __  |/ /  / /
 ___/ // // /_/ / /|  / ___ |/ /___   / /_/ // /  / /
/____/___/\____/_/ |_/_/  |_/_____/  /_____/___/ /_/
    '''
    print(art)
    command = {
        "data": "data.yaml",
        "model": "yolov8n.pt",
        "epochs": 100,
        "imgsz": 640,
        "name": "yolo_custom",
        # "batch": 16,
        # "workers": 8,
        # "device": 0,
        # "optimizer": "Adam",
        # "lr0": 0.01,
        # "cos_lr": True,
        # "warmup_epochs": 3,
        # "warmup_momentum": 0.8,
        # "warmup_bias_lr": 0.1,
        # "amp": True,
        # "overlap_mask": True,
        # "mask_ratio": 2,
        # "close_mosaic": 5, 
        # "val": True
    }
    command_str = [f"{key}={value}" for key, value in command.items()]
    command_str = ["yolo detect train"] + command_str
    command_str = " ".join(command_str)
    print(f"Command - {command_str}")
    input("\nEnter to exit")