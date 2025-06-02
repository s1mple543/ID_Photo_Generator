from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QComboBox, QSlider, QSpinBox, QGroupBox, 
                            QMessageBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import sys
import cv2
import os
from utils.image_processor import ImageProcessor
from utils.face_analyzer import FaceAnalyzer
from utils.model_trainer import ModelTrainer

class TrainingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, trainer, data_dir, model_path):
        super().__init__()
        self.trainer = trainer
        self.data_dir = data_dir
        self.model_path = model_path
    
    def run(self):
        try:
            self.progress_updated.emit(10, "正在加载数据集...")
            self.progress_updated.emit(30, "正在初始化模型...")
            
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            self.progress_updated.emit(50, f"开始训练 {model_name} 模型...")
            
            self.trainer.train_model(self.data_dir, self.model_path, epochs=20)
            
            self.progress_updated.emit(100, "训练完成!")
            self.finished.emit(True, f'模型 "{model_name}" 训练完成并已保存')
        except Exception as e:
            self.finished.emit(False, f"训练出错: {str(e)}")

class PhotoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能证件照生成器")
        self.setGeometry(100, 100, 1000, 700)
        
        self.image_processor = ImageProcessor()
        self.face_analyzer = FaceAnalyzer()
        self.model_trainer = ModelTrainer()
        self.current_image = None
        self.current_id_photo = None
        self.training_data_dir = None
        
        self.init_ui()
        self.check_models()
    
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.open_image)
        file_layout.addWidget(self.btn_open)
        
        self.btn_save = QPushButton("保存证件照")
        self.btn_save.clicked.connect(self.save_id_photo)
        self.btn_save.setEnabled(False)
        file_layout.addWidget(self.btn_save)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # 图像处理
        process_group = QGroupBox("图像处理")
        process_layout = QVBoxLayout()
        
        process_layout.addWidget(QLabel("缩放:"))
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(50, 200)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.apply_operations)
        process_layout.addWidget(self.slider_scale)
        
        process_layout.addWidget(QLabel("旋转角度:"))
        self.spin_rotate = QSpinBox()
        self.spin_rotate.setRange(-180, 180)
        self.spin_rotate.valueChanged.connect(self.apply_operations)
        process_layout.addWidget(self.spin_rotate)
        
        process_layout.addWidget(QLabel("滤镜:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["无", "灰度", "复古", "模糊", "边缘检测"])
        self.combo_filter.currentIndexChanged.connect(self.apply_operations)
        process_layout.addWidget(self.combo_filter)
        
        self.btn_process = QPushButton("检测人脸并生成证件照")
        self.btn_process.clicked.connect(self.process_faces)
        self.btn_process.setEnabled(False)
        process_layout.addWidget(self.btn_process)
        
        process_group.setLayout(process_layout)
        control_layout.addWidget(process_group)
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        model_layout.addWidget(QLabel("选择表情识别模型:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["自定义模型1", "自定义模型2"])
        self.combo_model.currentIndexChanged.connect(self.change_model)
        model_layout.addWidget(self.combo_model)
        
        model_layout.addWidget(QLabel("训练自定义模型:"))
        self.btn_select_data = QPushButton("选择训练数据集")
        self.btn_select_data.clicked.connect(self.select_training_data)
        model_layout.addWidget(self.btn_select_data)
        
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        model_layout.addWidget(self.train_progress)
        
        self.btn_train = QPushButton("训练模型")
        self.btn_train.clicked.connect(self.train_custom_model)
        model_layout.addWidget(self.btn_train)
        
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)
        
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        
        # 右侧图像显示
        image_group = QWidget()
        image_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        image_layout.addWidget(QLabel("原始/处理图像:"))
        image_layout.addWidget(self.image_label)
        
        self.id_photo_label = QLabel()
        self.id_photo_label.setAlignment(Qt.AlignCenter)
        self.id_photo_label.setMinimumSize(600, 400)
        image_layout.addWidget(QLabel("生成的证件照:"))
        image_layout.addWidget(self.id_photo_label)
        
        image_group.setLayout(image_layout)
        
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(image_group, 2)
        main_widget.setLayout(main_layout)
        
        self.setCentralWidget(main_widget)
        
        # 初始加载第一个模型
        self.change_model()
    
    def check_models(self):
        if not os.path.exists(self.face_analyzer.predictor_path):
            QMessageBox.warning(self, "缺少模型文件", 
                              "缺少人脸关键点检测模型，部分功能可能受限")
    
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", 
            "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.imagefile = file_name
            if self.current_image is not None:
                self.display_image(self.current_image, self.image_label)
                self.btn_process.setEnabled(True)
                self.id_photo_label.clear()
                self.btn_save.setEnabled(False)
            else:
                QMessageBox.warning(self, "错误", "无法加载图片文件")
    
    def apply_operations(self):
        if self.current_image is None:
            return
            
        scale = self.slider_scale.value() / 100.0
        angle = self.spin_rotate.value()
        filter_type = self.combo_filter.currentText()
        
        processed = self.current_image.copy()
        
        if scale != 1.0:
            processed = self.image_processor.resize(processed, width=int(processed.shape[1]*scale))
        
        if angle != 0:
            processed = self.image_processor.rotate(processed, angle)
        
        if filter_type != "无":
            filter_map = {
                "灰度": "grayscale",
                "复古": "sepia",
                "模糊": "blur",
                "边缘检测": "edge"
            }
            processed = self.image_processor.apply_filter(processed, filter_map[filter_type])
        
        self.display_image(processed, self.image_label)
    
    def process_faces(self):
        if self.current_image is None:
            return
            
        processed = self.current_image.copy()
        faces = self.face_analyzer.detect_faces(processed)
        
        if len(faces) == 0:
            self.image_label.setText("未检测到人脸")
            return
            
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(processed, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_roi = processed[y:y+h, x:x+w]
            emotion = self.face_analyzer.recognize_emotion(face_roi)
            
            cv2.putText(processed, emotion, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            id_photo = self.face_analyzer.generate_id_photo(self.imagefile, face, emotion)
            if id_photo is not None:
                self.display_image(id_photo, self.id_photo_label)
                self.current_id_photo = id_photo
                self.btn_save.setEnabled(True)
        
        self.display_image(processed, self.image_label)
    
    def save_id_photo(self):
        if not hasattr(self, 'current_id_photo') or self.current_id_photo is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存证件照", "", 
            "JPEG 图片 (*.jpg);;PNG 图片 (*.png)")
        
        if file_name:
            success = cv2.imwrite(file_name, self.current_id_photo)
            if success:
                QMessageBox.information(self, "保存成功", "证件照已保存")
            else:
                QMessageBox.warning(self, "保存失败", "无法保存图片文件")
    
    def change_model(self):
        model_name = self.combo_model.currentText()
        if self.face_analyzer.load_model(model_name):
            self.statusBar().showMessage(f"已切换到模型: {model_name}", 3000)
        else:
            QMessageBox.warning(self, "错误", f"无法加载模型 {model_name}")
    
    def select_training_data(self):
        data_dir = QFileDialog.getExistingDirectory(
            self, "选择训练数据集目录", 
            options=QFileDialog.ShowDirsOnly)
        
        if data_dir:
            self.training_data_dir = data_dir
            QMessageBox.information(
                self, "数据集选择", 
                f"已选择数据集目录: {data_dir}\n\n"
                "请确保目录包含以下子目录: angry, disgust, fear, happy, neutral, sad, surprise")
    
    def train_custom_model(self):
        if not hasattr(self, 'training_data_dir') or not self.training_data_dir:
            QMessageBox.warning(self, "错误", "请先选择训练数据集目录")
            return
        
        model_name = self.combo_model.currentText()
        if "预训练" in model_name:
            QMessageBox.warning(self, "错误", "不能覆盖预训练模型")
            return
        
        model_path = os.path.join(self.face_analyzer.models_dir, 
                                f"{model_name.lower().replace(' ', '_')}.pth")
        
        self.train_progress.setVisible(True)
        self.btn_train.setEnabled(False)
        self.train_progress.setValue(0)
        
        self.train_thread = TrainingThread(
            self.model_trainer, self.training_data_dir, model_path)
        self.train_thread.progress_updated.connect(self.update_train_progress)
        self.train_thread.finished.connect(self.on_training_finished)
        self.train_thread.start()
    
    def update_train_progress(self, value, message):
        self.train_progress.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_training_finished(self, success, message):
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "训练完成", message)
            self.change_model()  # 重新加载模型
        else:
            QMessageBox.critical(self, "训练失败", message)
    
    def display_image(self, image, label):
        if len(image.shape) == 2:  # 灰度图
            qimage = QImage(image.data, image.shape[1], image.shape[0], 
                           image.shape[1], QImage.Format_Grayscale8)
        else:  # 彩色图
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], 
                           rgb_image.shape[1]*3, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(
            label.width(), label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

def main():
    app = QApplication(sys.argv)
    window = PhotoApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()