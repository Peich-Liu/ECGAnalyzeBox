import sys
from PyQt5.QtWidgets import QApplication, QDialog
from UItest import Ui_Dialog  # 导入生成的 UI 类
class MyDialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)
        self.setupUi(self)  # 设置 UI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()  
    sys.exit(app.exec_())