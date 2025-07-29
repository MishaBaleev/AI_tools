import os
import shutil
from sklearn.model_selection import train_test_split
import ujson as json
import time

def valid_get_config() -> list or None:
    try: 
        with open("./config.json", "r") as config:
            classes = json.loads(config.read())["dataset_ratio"]
            return classes
    except ValueError as e:
        print(f"Error with config - {e}")
        return None

def create_yolo_structure(raw_path:str, output_path:str, test_dev_ratio, val_ratio) -> None:
    if not os.path.exists(os.path.join(raw_path, 'images')): raise FileNotFoundError(f"Папка images не найдена в {raw_path}")
    if not os.path.exists(os.path.join(raw_path, 'labels')): raise FileNotFoundError(f"Папка labels не найдена в {raw_path}")
    dirs = ['train', 'val', 'test_dev']
    dataset_path = os.path.join(output_path, f"dataset_{str(time.time())}")
    output_path = dataset_path
    for d in dirs:
        os.makedirs(os.path.join(dataset_path, 'images', d), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'labels', d), exist_ok=True)
    image_files = []
    label_files = []
    for subfolder in os.listdir(os.path.join(raw_path, 'images')):
        subfolder_path = os.path.join(raw_path, 'images', subfolder)
        if os.path.isdir(subfolder_path):
            for img in os.listdir(subfolder_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(subfolder_path, img)
                    label_path = os.path.join(raw_path, 'labels', subfolder, os.path.splitext(img)[0] + '.txt')
                    if os.path.exists(label_path):
                        image_files.append(img_path)
                        label_files.append(label_path)
                    else: print(f"Warning: Missing label for {img_path}")
    if not image_files: raise ValueError("Not a single image with markup was found.")
    try:
        train_val, test_dev, train_val_labels, test_dev_labels = train_test_split(image_files, label_files, test_size=test_dev_ratio, random_state=42)
        train, val, train_labels, val_labels = train_test_split(train_val, train_val_labels, test_size=val_ratio/(1-test_dev_ratio), random_state=42)
    except ValueError as e: raise ValueError(f"Separation error: {e}")

    def copy_files(files, labels, subset):
        for img_path, label_path in zip(files, labels):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            shutil.copy2(img_path, os.path.join(output_path, 'images', subset, img_name))
            shutil.copy2(label_path, os.path.join(output_path, 'labels', subset, label_name))

    try:
        copy_files(train, train_labels, 'train')
        copy_files(val, val_labels, 'val')
        copy_files(test_dev, test_dev_labels, 'test_dev')
    except Exception as e: raise RuntimeError(f"Copy error: {e}")

    print(f"Successfully created:\n"
          f"Train: {len(train)} images\n"
          f"Val: {len(val)} images\n"
          f"Test_dev: {len(test_dev)} images")

if __name__ == "__main__":
    art = '''
   _____ ___________   _____    __       ____  __________
  / ___//  _/ ____/ | / /   |  / /      / __ )/  _/_  __/
  \__ \ / // / __/  |/ / /| | / /      / __  |/ /  / /
 ___/ // // /_/ / /|  / ___ |/ /___   / /_/ // /  / /
/____/___/\____/_/ |_/_/  |_/_____/  /_____/___/ /_/
    '''
    print(art)
    try:
        ratios = valid_get_config()
        create_yolo_structure(
            raw_path="raw_dataset",
            output_path='datasets',
            test_dev_ratio=ratios["test"],
            val_ratio=ratios["val"]
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Check the folder structure and the presence of files, as well as the configuration file.")
    
    input("\nEnter to exit")