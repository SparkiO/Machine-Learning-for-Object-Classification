from PyQt5 import QtWidgets, QtGui, QtCore
from Ui_MainWindow import Ui_MainWindow
from PyQt5.QtCore import QSize
import sys

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.window = Ui_MainWindow()
        self.window.setupUi(self)

        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_controller(self, controller):
        self.controller = controller

    def checkbox_toggled(self, int):
        self.controller.model_enablement_changed()

    def list_index_moved(self, next_item, previous_item):
        path = next_item.data(QtCore.Qt.UserRole)
        self.controller.image_selected(path)

    def is_checked(self, name):
        return getattr(self.window, "{}_check".format(name)).isChecked()

    def clear_graph(self, name):
        getattr(self.window, "{}_img".format(name)).setText("?")

    def show_graph(self, path, name):
        pixmap = QtGui.QPixmap(path).scaled(260,195)
        label = getattr(self.window, "{}_img".format(name))
        label.setPixmap(pixmap)

    def show_list(self, paths, name):
        for path in paths:
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, path)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item.setIcon(icon)
            getattr(self.window, "{}_list".format(name)).addItem(item)

    def load_data(self, class_name, file_index):
        layer_folder = "layers"
        map_folder = "saliency_maps"

        index = ""
        for c in file_index:
            if (c != "_"):
                index += c
            else:
                break
                
        
        #layer loading:
        path = "data/" + class_name + "/" + layer_folder + "/"
        google_first = path + index + "GoogLeNet_first_256.jpg"
        pixmap = QtGui.QPixmap(google_first).scaled(250, 155)
        self.window.GoogLeNet_layers_first.setPixmap(pixmap)
        self.window.GoogLeNet_layers_first.setAlignment(QtCore.Qt.AlignCenter)
        
        google_last = path + index + "GoogLeNet_last_256.jpg"
        pixmap = QtGui.QPixmap(google_last).scaled(250, 155)
        self.window.GoogLeNet_layers_last.setPixmap(pixmap)
        self.window.GoogLeNet_layers_last.setAlignment(QtCore.Qt.AlignCenter)
        
        vgg_first = path + index + "VGG_first_256.jpg"
        pixmap = QtGui.QPixmap(vgg_first).scaled(250, 155)
        self.window.VGG_layers_first.setPixmap(pixmap)
        self.window.VGG_layers_first.setAlignment(QtCore.Qt.AlignCenter)
        
        vgg_last = path + index + "VGG_last_256.jpg"
        pixmap = QtGui.QPixmap(vgg_last).scaled(250, 155)
        self.window.VGG_layers_last.setPixmap(pixmap)
        self.window.VGG_layers_last.setAlignment(QtCore.Qt.AlignCenter)

        #saliency maps:
        path = "data/" + class_name + "/" + map_folder + "/"
        google_first = path + index + "GoogLeNet_256.jpg"
        pixmap = QtGui.QPixmap(google_first).scaled(250, 210)
        self.window.GoogLeNet_sal.setPixmap(pixmap)
        self.window.GoogLeNet_sal.setAlignment(QtCore.Qt.AlignCenter)
        
        google_first = path + index + "GoogLeNet_256_smoothgrad.jpg"
        pixmap = QtGui.QPixmap(google_first).scaled(250, 210)
        self.window.GoogLeNet_smoothgrad.setPixmap(pixmap)
        self.window.GoogLeNet_smoothgrad.setAlignment(QtCore.Qt.AlignCenter)


        vgg = path + index + "VGG_256.jpg"
        pixmap = QtGui.QPixmap(vgg).scaled(250, 210)
        self.window.VGG_sal.setPixmap(pixmap)
        self.window.VGG_sal.setAlignment(QtCore.Qt.AlignCenter)
        
        google_first = path + index + "VGG_256_smoothgrad.jpg"
        pixmap = QtGui.QPixmap(google_first).scaled(250, 210)
        self.window.VGG_smoothgrad.setPixmap(pixmap)
        self.window.VGG_smoothgrad.setAlignment(QtCore.Qt.AlignCenter)

        name = "dense_layer_vis/google_" + class_name + ".gif"
        movie = QtGui.QMovie(name)
        movie.setScaledSize(QSize(260, 195))
        self.window.GoogLeNet_dense.setMovie(movie)
        movie.start()

        name = "dense_layer_vis/vgg_" + class_name + ".gif"
        movie = QtGui.QMovie(name)
        movie.setScaledSize(QSize(260, 195))
        self.window.VGG_dense.setMovie(movie)
        movie.start()

    def show_image(self, path):
        pixmap = QtGui.QPixmap(path).scaled(448, 448)
        self.window.main_image.setPixmap(pixmap)
        self.window.main_image.setAlignment(QtCore.Qt.AlignCenter)
        
        filename = ""
        for c in path[::-1]:
            if (c != "\\"):
                filename += c
            else:
                break
        filename = filename[::-1]

        if "dog" in path:
            self.load_data("dog", filename)
        elif "bird" in path:
            self.load_data("bird", filename)
        elif "berry" in path:
            self.load_data("berry", filename)
        elif "flower" in path:
            self.load_data("flower", filename)
        elif "other" in path:
            self.load_data("other", filename)
    
    def check_all_models(self, load = True, names = []):
        if (load):
            self.window.AlexNet_check.setChecked(True)
            self.window.ZFNet_check.setChecked(True)
            self.window.VGG_check.setChecked(True)
            self.window.GoogLeNet_check.setChecked(True)
            self.window.ResNet_check.setChecked(True)
            self.window.MNaSNet_check.setChecked(True)

qt_app = None
def prepare_ui():
    global qt_app
    qt_app = QtWidgets.QApplication([])
    qt_app.setStyle('fusion')

def loop_ui():
    global qt_app
    sys.exit(qt_app.exec())
    