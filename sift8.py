import os
import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans
from tkinter import Tk, Label, Button, Canvas
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, ImageOps

# 定义余弦相似度计算函数
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 提取图像特征
def extract_features(image_path, vector_size=32):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f'Failed to load image at "{image_path}"')
        return None, None
    
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        if len(keypoints) < vector_size:
            print(f'Not enough features detected in image at "{image_path}"')
            return np.zeros(vector_size * 128), np.zeros(16 * 16 * 16) 

        vocabulary, _ = kmeans(descriptors, vector_size)

        hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return vocabulary.flatten(), hist
    except cv2.error as e:
        print(f'Error: {e}')
        return np.zeros(vector_size * 128), np.zeros(16 * 16 * 16) 

# 以图搜图
def search_image(input_image_path, dataset_path, top_k=10):
    sift_input, hist_input = extract_features(input_image_path)

    if sift_input is None or hist_input is None:
        print("Failed to extract features from the input image.")
        return []

    sift_dataset = []
    hist_dataset = []
    dataset_images = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            sift_features, hist_features = extract_features(image_path)
            
            if sift_features is None or hist_features is None:
                print(f"Failed to extract features from image at '{image_path}'.")
                continue
            
            sift_dataset.append(sift_features)
            hist_dataset.append(hist_features)
            dataset_images.append(image_path)

    # sift_distances = np.linalg.norm(sift_dataset - sift_input, axis=1)
    # hist_distances = np.linalg.norm(hist_dataset - hist_input, axis=1)

    # 计算余弦相似度
    sift_similarities = np.array([cosine_similarity(sift_input, sift) for sift in sift_dataset])
    hist_similarities = np.array([cosine_similarity(hist_input, hist) for hist in hist_dataset])
    weights_sift = 0.8
    weights_hist = 0.2
    #total_distances = weights_sift * sift_distances + weights_hist * hist_distances
    total_similarities = weights_sift * sift_similarities + weights_hist * hist_similarities
    # nearest_neighbors_indices = np.argsort(total_distances)[:top_k]
    # nearest_neighbors_images = [dataset_images[i] for i in nearest_neighbors_indices]
    # 注意，这里我们要找的是最相似的图像，所以应该取相似度最大的top_k个图像，而不是最小的
    nearest_neighbors_indices = np.argsort(total_similarities)[-top_k:]
    nearest_neighbors_images = [dataset_images[i] for i in nearest_neighbors_indices]
    
    return nearest_neighbors_images

# 创建GUI应用
class Application(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title("Image Search")
        self.geometry("800x600")

        self.input_image_label = Label(self, text="Input Image")
        self.input_image_label.grid(row=0, column=0)

        self.input_image_canvas = Canvas(self, width=200, height=200)
        self.input_image_canvas.grid(row=1, column=0)

        self.select_button = Button(self, text="Select Image", command=self.select_image)
        self.select_button.grid(row=1, column=1)
        self.result_images_canvas = [Canvas(self, width=200, height=200) for _ in range(10)]
        for i, canvas in enumerate(self.result_images_canvas):
            canvas.grid(row=2+i//5, column=i%5)

    # 选择图像
    def select_image(self):
        file_path = askopenfilename()
        if file_path:
            input_image = Image.open(file_path)
            input_image = ImageOps.fit(input_image, (200, 200), Image.ANTIALIAS)
            input_image_tk = ImageTk.PhotoImage(input_image)
            self.input_image_canvas.create_image(100, 100, anchor='center', image=input_image_tk)
            self.input_image_canvas.image = input_image_tk
            result_images_path = search_image(file_path, dataset_path, top_k=10)
            for canvas, image_path in zip(self.result_images_canvas, result_images_path):
                result_image = Image.open(image_path)
                result_image = ImageOps.fit(result_image, (200, 200), Image.ANTIALIAS)
                result_image_tk = ImageTk.PhotoImage(result_image)
                canvas.create_image(100, 100, anchor='center', image=result_image_tk)
                canvas.image = result_image_tk

if __name__ == "__main__":
    dataset_path = "D:\VSCforPython\duomeiti\corel"
    app = Application()
    app.mainloop()