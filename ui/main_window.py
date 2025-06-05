from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QComboBox, QSlider, QSpinBox, QGroupBox, 
                            QMessageBox, QProgressBar,)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QEvent, QTimer
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
        self.setGeometry(100, 100, 1500, 800)
        
        self.image_processor = ImageProcessor()
        self.face_analyzer = FaceAnalyzer()
        self.model_trainer = ModelTrainer()
        self.current_image = None
        self.current_id_photo = None
        self.training_data_dir = None
        self.resize_event_connected = False
        self.installEventFilter(self)
        
        self.init_ui()
        self.check_models()
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            if not self.resize_event_connected:
                QTimer.singleShot(100, self.adjust_image_sizes)
                self.resize_event_connected = True
            else:
                self.resize_event_connected = False
        return super().eventFilter(obj, event)

    def adjust_image_sizes(self):
        # 统一调整三个图像显示区域的大小
        min_size = min(
            self.original_image.width(), 
            self.original_image.height()
        )
        new_size = min(400, min_size)  # 不超过400x400
        
        for label in [self.original_image, self.processed_image, self.idphoto_image]:
            label.setFixedSize(new_size, new_size)
        
        # 重绘当前图像
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.display_image(self.current_image, self.original_image)
        if hasattr(self, 'current_processed_image') and self.current_processed_image is not None:
            self.display_image(self.current_processed_image, self.processed_image)
        if hasattr(self, 'current_id_photo') and self.current_id_photo is not None:
            self.display_image(self.current_id_photo, self.idphoto_image)
    
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
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        
        # 图像处理
        process_group = QGroupBox("图像处理")
        process_layout = QVBoxLayout()
        
        process_layout.addWidget(QLabel("旋转角度:"))
        self.spin_rotate = QSpinBox()
        self.spin_rotate.setRange(-180, 180)
        self.spin_rotate.valueChanged.connect(self.apply_operations)
        process_layout.addWidget(self.spin_rotate)
        
        process_layout.addWidget(QLabel("平移（左右）:"))
        self.spin_translate_x = QSpinBox()
        self.spin_translate_x.setRange(-500, 500)
        self.spin_translate_x.valueChanged.connect(self.apply_operations)
        process_layout.addWidget(self.spin_translate_x)
        
        process_layout.addWidget(QLabel("平移（上下）:"))
        self.spin_translate_y = QSpinBox()
        self.spin_translate_y.setRange(-500, 500)
        self.spin_translate_y.valueChanged.connect(self.apply_operations)
        process_layout.addWidget(self.spin_translate_y)
        
        process_layout.addWidget(QLabel("镜像:"))
        self.combo_mirror = QComboBox()
        self.combo_mirror.addItems(["无", "水平镜像", "垂直镜像"])
        self.combo_mirror.currentIndexChanged.connect(self.apply_operations)
        process_layout.addWidget(self.combo_mirror)

        process_layout.addWidget(QLabel("仿射变换:"))
        self.combo_AT = QComboBox()
        self.combo_AT.addItems(["无", "仿射变换"])
        self.combo_AT.currentIndexChanged.connect(self.apply_operations)
        process_layout.addWidget(self.combo_AT)

        process_layout.addWidget(QLabel("空域的平滑和锐化:"))
        self.combo_space = QComboBox()
        self.combo_space.addItems(["无", "直方图绘制", "直方图均衡化", "灰度变换（对数变换）", "平滑滤波", "锐化滤波"])
        self.combo_space.currentIndexChanged.connect(self.apply_operations)
        process_layout.addWidget(self.combo_space)
        
        process_layout.addWidget(QLabel("频域的平滑和锐化:"))
        self.combo_frequency = QComboBox()
        self.combo_frequency.addItems(["无", "低通滤波", "高通滤波", "带通滤波"])
        self.combo_frequency.currentIndexChanged.connect(self.apply_operations)
        process_layout.addWidget(self.combo_frequency)
        
        process_layout.addWidget(QLabel("形态学操作:"))
        self.combo_Morphological = QComboBox()
        self.combo_Morphological.addItems(["无", "开运算", "闭运算"])
        self.combo_Morphological.currentIndexChanged.connect(self.apply_operations)
        process_layout.addWidget(self.combo_Morphological)
        
        process_layout.addWidget(QLabel("边缘检测:"))
        self.combo_edge = QComboBox()
        self.combo_edge.addItems(["无", "边缘检测"])
        self.combo_edge.currentIndexChanged.connect(self.apply_operations)
        process_layout.addWidget(self.combo_edge)
        
        process_layout.addWidget(QLabel("证件照背景颜色:"))
        self.combo_bg_color = QComboBox()
        self.combo_bg_color.addItems(["白色", "红色", "蓝色"])
        process_layout.addWidget(self.combo_bg_color)
        
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
        image_layout = QHBoxLayout()
        # 创建固定大小的图像显示区域
        def create_image_box(title, border_color="gray"):
            group = QVBoxLayout()
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            image = QLabel()
            image.setAlignment(Qt.AlignCenter)
            image.setFixedSize(400, 400)  # 固定尺寸
            image.setStyleSheet(f"""
                border: 2px solid {border_color};
                background-color: #f0f0f0;
                qproperty-alignment: 'AlignCenter';
            """)
            group.addWidget(label)
            group.addWidget(image)
            return group, image
        
        # 原始图像组
        original_group, self.original_image = create_image_box("原始图像", "green")
        self.btn_original = QPushButton("对齐用")
        original_group.addWidget(self.btn_original)
        
        # 处理后图像组
        processed_group, self.processed_image = create_image_box("处理后图像", "orange")
        self.btn_save_processed = QPushButton("保存处理图像")
        self.btn_save_processed.clicked.connect(self.save_processed_image)
        processed_group.addWidget(self.btn_save_processed)
        
        # 证件照组
        idphoto_group, self.idphoto_image = create_image_box("证件照", "blue")
        self.btn_save_idphoto = QPushButton("保存证件照")
        self.btn_save_idphoto.clicked.connect(self.save_id_photo)
        idphoto_group.addWidget(self.btn_save_idphoto)

        # 添加到主布局
        image_layout.addLayout(original_group)
        image_layout.addLayout(processed_group)
        image_layout.addLayout(idphoto_group)
        image_group.setLayout(image_layout)

        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(image_group, 3)
        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)

        # 初始化状态
        self.btn_save_processed.setEnabled(False)
        self.btn_save_idphoto.setEnabled(False)
        self.current_processed_image = None
        self.current_id_photo = None
        
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
                # 检查图像尺寸是否过大
                h, w = self.current_image.shape[:2]
                if w > 1000 or h > 1000:
                    self.current_image = cv2.resize(
                        self.current_image, 
                        (1000, int(1000*h/w)) if w > h else (int(1000*w/h), 1000))
                
                self.display_image(self.current_image, self.original_image)
                self.processed_image.clear()
                self.idphoto_image.clear()
                self.btn_process.setEnabled(True)
                self.btn_save_processed.setEnabled(False)
                self.btn_save_idphoto.setEnabled(False)
            else:
                QMessageBox.warning(self, "错误", "无法加载图片文件")
    
    def apply_operations(self):
        if self.current_image is None:
            return
        
        angle = self.spin_rotate.value()
        x = self.spin_translate_x.value()
        y = self.spin_translate_y.value()
        mirror_type = self.combo_mirror.currentText()
        affine_type = self.combo_AT.currentText()
        space_type = self.combo_space.currentText()
        frequency_type = self.combo_frequency.currentText()
        morphological_type = self.combo_Morphological.currentText()
        edge_type = self.combo_edge.currentText()
        
        processed = self.current_image.copy()

        # Apply transformations in a logical order
        if angle != 0:
            processed = self.image_processor.rotate(processed, angle)
            
        if x != 0 or y != 0:
            processed = self.image_processor.translate(processed, x, y)

        if mirror_type != "无":
            mirror_map = {
                "水平镜像": "horizontal_mirroring",
                "垂直镜像": "vertical_mirroring"
            }
            processed = self.image_processor.apply_mirror(processed, mirror_map[mirror_type])

        if affine_type == "仿射变换":
            processed = self.image_processor.apply_affine_transform(processed)

        if space_type != "无":
            space_map = {
                "直方图绘制": "histogram_drawing",
                "直方图均衡化": "histogram_equalization",
                "灰度变换（对数变换）": "grayscale_transformation",
                "平滑滤波": "smoothing_filtering",
                "锐化滤波": "sharpen_filtering"
            }
            processed = self.image_processor.apply_space(processed, space_map[space_type])

        if frequency_type != "无":
            frequency_map = {
                "低通滤波": "lowpass_filtering",
                "高通滤波": "highpass_filtering",
                "带通滤波": "bandpass_filtering"
            }
            processed = self.image_processor.apply_frequency(processed, frequency_map[frequency_type])

        if morphological_type != "无":
            morphological_map = {
                "开运算": "open_operation",
                "闭运算": "close_operation"
            }
            processed = self.image_processor.apply_Morphological(processed, morphological_map[morphological_type])

        if edge_type == "边缘检测":
            processed = self.image_processor.apply_edge_detection(processed)
        
        self.current_processed_image = processed
        self.display_image(processed, self.processed_image)
        self.btn_save_processed.setEnabled(True)

    def process_faces(self):
        if self.current_image is None:
            return
            
        processed = self.current_image.copy()
        faces = self.face_analyzer.detect_faces(processed)
        
        if len(faces) == 0:
            self.processed_image.setText("未检测到人脸")
            return
            
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(processed, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_roi = processed[y:y+h, x:x+w]
            emotion = self.face_analyzer.recognize_emotion(face_roi)
            
            cv2.putText(processed, emotion, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            bg_color = self.combo_bg_color.currentText()
            id_photo = self.face_analyzer.generate_id_photo(self.imagefile, face, emotion, bg_color)
            if id_photo is not None:
                self.display_image(id_photo, self.idphoto_image)
                self.current_id_photo = id_photo
                self.btn_save_idphoto.setEnabled(True)
        
        self.current_processed_image = processed
        self.display_image(processed, self.processed_image)
        self.btn_save_processed.setEnabled(True)
    
    def save_processed_image(self):
        if self.current_processed_image is None:
            QMessageBox.warning(self, "错误", "没有可保存的处理后图像")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存处理后的图像", "处理后图像", 
            "JPEG 图片 (*.jpg);;PNG 图片 (*.png)")
        
        if file_name:
            success = cv2.imwrite(file_name, self.current_processed_image)
            if success:
                QMessageBox.information(self, "保存成功", f"处理后图像已保存到:\n{file_name}")
            else:
                QMessageBox.warning(self, "保存失败", "无法保存图片文件")
    
    def save_id_photo(self):
        if not hasattr(self, 'current_id_photo') or self.current_id_photo is None:
            QMessageBox.warning(self, "错误", "没有可保存的证件照")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存证件照", "证件照", 
            "JPEG 图片 (*.jpg);;PNG 图片 (*.png)")

        if file_name:
            success = cv2.imwrite(file_name, self.current_id_photo)
            if success:
                QMessageBox.information(self, "保存成功", f"证件照已保存到:\n{file_name}")
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
        if image is None:
            label.clear()
            label.setText("无图像")
            return

        # 转换图像格式
        if len(image.shape) == 2:  # 灰度图
            qimage = QImage(image.data, image.shape[1], image.shape[0], 
                           image.shape[1], QImage.Format_Grayscale8)
        else:  # 彩色图
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], 
                           rgb_image.shape[1]*3, QImage.Format_RGB888)

        # 创建Pixmap并保持原始尺寸
        pixmap = QPixmap.fromImage(qimage)

        # 如果图像大于显示区域，按比例缩小
        if pixmap.width() > label.width() or pixmap.height() > label.height():
            pixmap = pixmap.scaled(
                label.width(), label.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

def main():
    app = QApplication(sys.argv)
    window = PhotoApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()