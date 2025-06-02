import sys
from ui.main_window import PhotoApp
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoApp()
    window.show()
    sys.exit(app.exec_())