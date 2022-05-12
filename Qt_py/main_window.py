import sys
import os
import impred_rc
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5 import uic,QtWidgets,QtCore
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import  QApplication, QFileDialog
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
#for tensor based operations
from tensorflow.keras.utils import normalize
# visualisation de la model pour bien comprendre ce qui va se passé
from tensorflow.keras.utils import plot_model
from pathlib import Path






# qtui = "age_pred.ui"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
qtui = os.path.join(CURRENT_DIR, "age_pred.ui")


Ui_MainWindow, QtBaseClass = uic.loadUiType(qtui)
class MyAppp(QtWidgets.QMainWindow,Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.ajimg.clicked.connect(self.open)
        # self.Bt.clicked.connect(self.SequentialModel)
        # self.Bt2.clicked.connect(self.Fermer)
        # self.Bt3.clicked.connect(self.FonctionalModel)
        # self.path_image = 'images'
        # self.SequentialModel()

    def Fermer(self):
        self.close()

    def open(self):
        path= QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
        img = Image.open(path[0])
        # print(img)
       
        if path != ('', ''):
            print(path[0])
            tableauPixels = np.array(img)
            print(tableauPixels)
            nouvellePhoto = Image.fromarray(tableauPixels)
            print(nouvellePhoto)
            # print(CURRENT_DIR)
            # print(os.path.join(CURRENT_DIR, "image.jpeg"))
            nouvellePhoto.save(os.path.join(CURRENT_DIR, "image.jpeg"), "JPEG")
            
            self.impred.setPixmap(QPixmap(os.path.join(CURRENT_DIR, "image.jpeg")))
            # load a model => charger le model model.h5
            model = load_model(os.path.join(CURRENT_DIR, "age_prediction_model_ml.h5"))
            print('pred')
            test_pic = cv2.imread(r'C:\Users\Republic Of Computer\Desktop\Master cours et TD\IA\Projet\Age_prediction\Qt_py\image.jpeg')
            # print('pred')
            image = cv2.cvtColor(test_pic,cv2.COLOR_BGR2RGB)
            test_pic = cv2.resize(image,(128,128))
            test_pic = test_pic.reshape((1,128,128,3))
            pred = model.predict(test_pic)
            result = int(pred)
            self.result.setText("L'age est à peu près sde "+str(result))

            print('Sequential Model')
            return result
            
        



    def SequentialModel(self):

        # load a model => charger le model model.h5
        model = load_model(os.path.join(CURRENT_DIR, "age_prediction_model_ml.h5"))
        print('pred')
        test_pic = cv2.imread(r'C:\Users\Republic Of Computer\Desktop\Master cours et TD\IA\Projet\Age_prediction\Qt_py\image.jpeg')
        # print('pred')
        image = cv2.cvtColor(test_pic,cv2.COLOR_BGR2RGB)
        test_pic = cv2.resize(image,(128,128))
        test_pic = test_pic.reshape((1,128,128,3))
        pred = model.predict(test_pic)
        result = int(pred)
        self.result.setText("L'age est à peu près de"+str(result))

        print('Sequential Model')
        return result

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyAppp()
    gui.show()
    sys.exit(app.exec_())


