from Model import BaseModel
from Plotting import Plot
from Ui_View import Window, prepare_ui, loop_ui

import importlib
modellib = importlib.import_module("Model")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DATA_PATH = "data/"
MODEL_PATH = "model/"
MODEL_NAMES = ["AlexNet", "ZFNet", "VGG", "GoogLeNet", "ResNet", "MNaSNet"]

class Controller:
    def __init__(self, ui):
        self.ui = ui
        self.models = {}
        self.img_selected_path = None

    def load_images(self, path):
        labels = next(os.walk(path))[1]
        for label in labels:
            paths = []
            for file in os.listdir(os.path.join(path, label)):
                paths.append(os.path.join(path, label, file))
            self.ui.show_list(paths, label)

    def refresh_graphs(self, to_refresh=None, to_clear=None):
        for model_name, model in self.models.items():
            if to_refresh is None or model_name in to_refresh:
                data = model.predict(self.img_selected_path)
                sample_name = "{}-{}".format( \
                    os.path.basename(os.path.dirname(self.img_selected_path)), \
                        os.path.splitext(os.path.basename(self.img_selected_path))[0])
                self.ui.show_graph(Plot(data, sample_name, model_name).plot(), model_name)
        if to_clear is not None:
            for model_name in to_clear:
                self.ui.clear_graph(model_name)

    def model_enablement_changed(self):
        for model_name in MODEL_NAMES:
            if self.ui.is_checked(model_name):
                if not model_name in self.models:
                    path = "{}{}.h5".format(MODEL_PATH, model_name)
                    try: model = getattr(modellib, model_name)(path)
                    except AttributeError: model = BaseModel(path)
                    self.models[model_name] = model
                    if self.img_selected_path is not None: 
                        self.refresh_graphs(to_refresh = model_name)
            else:
                if model_name in self.models:
                    del self.models[model_name]
                    if self.img_selected_path is not None: 
                        self.refresh_graphs(to_clear = model_name)

    def image_selected(self, path):
        self.img_selected_path = path
        self.ui.show_image(path)
        self.refresh_graphs()
        

def main():
    prepare_ui()
    ui = Window()
    controller = Controller(ui)
    ui.set_controller(controller)
    controller.load_images(DATA_PATH)
    ui.check_all_models(True, MODEL_NAMES)

    ui.show()
    loop_ui()

main()
