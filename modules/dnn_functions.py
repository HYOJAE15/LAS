from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QGraphicsScene

from .ui_main import Ui_MainWindow
from .ui_sam_window import Ui_SAMWindow
from .ui_functions import UIFunctions
from .app_settings import Settings

from .utils import cvtPixmapToArray

# from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
# import pydensecrf.densecrf as dcrf 

import numpy as np

from segment_anything import sam_model_registry, SamPredictor

class SAMWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_SAMWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

        # add qlabels to scroll area

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def setScene(self, pixmap, color_pixmap, scale=1.0):
        """
        Set the scene of the image
        Args:
            pixmap (QPixmap): The pixmap of the image.
            color_pixmap (QPixmap): The pixmap of the color image.
            scale (float): The scale of the scene.
        """
        self.scene = QGraphicsScene()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.color_pixmap_item = self.scene.addPixmap(color_pixmap)
        self.ui.graphicsView.setScene(self.scene)
        self.scaleScene(scale=scale)

    def scaleScene(self, scale=1.0):
        """
        Scale the scene
        Args:
            scale (float): The scale of the scene.
        """
        self.ui.graphicsView.setFixedSize(scale * self.pixmap_item.pixmap().size())
        self.ui.graphicsView.fitInView(self.pixmap_item)



class DNNFunctions(object):
    def __init__(self):

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        self.SAMWindow = SAMWindow()

        ######################### 
        # Semantic Segmentation #
        #########################
        
        # Segment Anything
        self.sam_checkpoint = 'dnn/checkpoints/sam_vit_h_4b8939.pth'

        ##############
        # Attributes #
        ##############
        
        self.scale = 1.0

    
    def load_sam(self, checkpoint, mode='default'):
        """
        Load the sam model
        Args:
            mode (str): The mode of the sam model.
        """
        self.sam_model = sam_model_registry[mode](checkpoint=checkpoint)
        self.sam_model.to(device='cuda:0')
        self.sam_predictor = SamPredictor(self.sam_model)
        self.set_sam_image()
        self.sam_status = True
        self.mmseg_status = False

        
    def set_sam_image(self):
        image = cvtPixmapToArray(self.pixmap)
        image = image[:, :, :3]
        
        self.sam_predictor.set_image(image)


    # @staticmethod
    # def applyDenseCRF(img, label, num_iter=3):
    #     """
    #     Apply DenseCRF to the image and label

    #     Args:
    #         img (np.ndarray): The image to be processed.
    #         label (np.ndarray): The label to be processed.
    #         num_iter (int): The number of iterations.

    #     Returns:
    #         label (np.ndarray): The processed label.
    #     """
    #     num_labels = np.max(label) + 1

    #     d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], num_labels)

    #     U = unary_from_labels(label, num_labels, gt_prob=0.7, zero_unsure=False)

    #     d.setUnaryEnergy(U)

    #     feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    #     d.addPairwiseEnergy(feats, compat=3,
    #                         kernel=dcrf.DIAG_KERNEL,
    #                         normalization=dcrf.NORMALIZE_SYMMETRIC)

    #     # This creates the color-dependent features and then add them to the CRF
    #     feats = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13),
    #                                         img=img, chdim=2)
    #     d.addPairwiseEnergy(feats, compat=10,
    #                         kernel=dcrf.DIAG_KERNEL,
    #                         normalization=dcrf.NORMALIZE_SYMMETRIC)

    #     Q = d.inference(num_iter)

    #     MAP = np.argmax(Q, axis=0)

    #     return MAP.reshape((img.shape[0], img.shape[1]))

    
    @staticmethod
    def cvtRGBATORGB(img):
        """Convert a RGBA image to a RGB image
        Args:
            img (np.ndarray): The image to be converted.

        Returns:
            img (np.ndarray): The converted image.
        
        """
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    

    


    