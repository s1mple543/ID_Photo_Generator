import dlib
import torch
import os
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from .model_trainer import EmotionCNN
from rembg import remove
import io
from io import BytesIO

class FaceAnalyzer:
    def __init__(self, models_dir="demo2\models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 初始化人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = os.path.join(self.models_dir, "shape_predictor_68_face_landmarks.dat")
        if os.path.exists(self.predictor_path):
            self.predictor = dlib.shape_predictor(self.predictor_path)
        else:
            self.predictor = None
        
        # 模型配置
        self.model_choices = {
            "自定义模型1": os.path.join(self.models_dir, "custom_model1.pth"),
            "自定义模型2": os.path.join(self.models_dir, "custom_model2.pth")
        }
        self.current_model_name = None
        self.emotion_model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def load_model(self, model_name):
        model_path = self.model_choices.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"模型 {model_name} 不存在于 {model_path}")
            return False
        
        self.current_model_name = model_name
        self.emotion_model = EmotionCNN().to(torch.device("cpu"))
        self.emotion_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.emotion_model.eval()
        return True
    
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces
    
    def recognize_emotion(self, face_roi):
        if self.emotion_model is None:
            return "Unknown"
        
        input_tensor = self.transform(face_roi).unsqueeze(0)
        with torch.no_grad():
            outputs = self.emotion_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = self.emotion_labels[predicted.item()]
        return emotion
    
    def generate_id_photo(self, image, face, emotion):
        if emotion not in ['Neutral', 'Happy']:
            return None
            
        if self.predictor is None:
            return None
            
        with open(image, "rb") as f:
            input_img = f.read()
        result = remove(input_img)
        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        bg_color = (255, 255, 255)
        bg = Image.new("RGBA", fg.size, bg_color + (255,))
        final = Image.alpha_composite(bg, fg)
        final_byte_array = io.BytesIO()
        final.save(final_byte_array, format="PNG")
        final_byte_array.seek(0)
        final = cv2.imdecode(np.frombuffer(final_byte_array.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        return final