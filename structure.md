/PhotoApp
│   main.py                  # 主程序入口
│   build.spec               # PyInstaller打包配置
│   requirements.txt         # 依赖库
│
├───models                   # 模型目录
│       pretrained_emotion_model.pth
│       custom_model1.pth
│       custom_model2.pth
│       shape_predictor_68_face_landmarks.dat
│
├───utils                    # 工具类
│       image_processor.py
│       face_analyzer.py
│       model_trainer.py
│
└───ui                       # 界面相关
        main_window.py