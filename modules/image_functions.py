import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt
from PySide6.QtGui import  QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QFileSystemModel, QGraphicsScene, QFileDialog
) 

from .ui_main import Ui_MainWindow
from .ui_functions import UIFunctions
from .ui_brush_menu import Ui_BrushMenu
from .ui_erase_menu import Ui_EraseMenu
from .app_settings import Settings
from .dnn_functions import DNNFunctions

from .utils import imread, cvtArrayToQImage, cvtPixmapToArray, convertLabelToColorMap, blendImageWithColorMap, getPrompt
from .utils_img import (
    getScaledPoint, getScaledPoint_mmdet, getCoordBTWTwoPoints, applyBrushSize, readImageToPixmap, histEqualization_gr, histEqualization_hsv, histEqualization_ycc
)

from modules.utils import imwrite_colormap

import skimage.measure
import skimage.filters

import copy

class BrushMenuWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_BrushMenu()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

class EraseMenuWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_EraseMenu()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


class ImageFunctions(DNNFunctions):
    def __init__(self):
        DNNFunctions.__init__(self)

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        global mainWidgets
        mainWidgets = self.ui
            
        mainWidgets.treeView.clicked.connect(self.openImage)
        self.fileModel = QFileSystemModel()
        self.alpha = 50
        self.scale = 1
        self.oldPos = None
        self.brush_class = 1
        # self.AltKey = False
        
        mainWidgets.mainImageViewer.mouseMoveEvent = self._mouseMoveEvent
        mainWidgets.mainImageViewer.mousePressEvent = self._mousePressPoint
        mainWidgets.mainImageViewer.mouseReleaseEvent = self._mouseReleasePoint
        # mainWidgets.mainImageViewer.mouseDoubleClickEvent = self._mouseDoubleClick

        
        mainWidgets.addImageButton.clicked.connect(self.addNewImage)
        mainWidgets.deleteImageButton.clicked.connect(self.deleteImage)

        
        self.sam_mode = False


        """
        Brush Tool
        """
        mainWidgets.brushButton.clicked.connect(self.openBrushMenu)

        self.BrushMenu = BrushMenuWindow()
        self.BrushMenu.ui.brushSizeSlider.valueChanged.connect(self.changeBrushSize)

        self.use_brush = False
        
        """
        Erase Tool
        """
        mainWidgets.eraseButton.clicked.connect(self.openEraseMenu)

        self.EraseMenu = EraseMenuWindow()
        self.EraseMenu.ui.eraseSizeSlider.valueChanged.connect(self.changeEraseSize)
        
        self.use_erase = False

        """
        Autolabel Tool
        """
        mainWidgets.autoLabelButton.clicked.connect(self.checkAutoLabelButton)
        self.use_autolabel = False

        # mainWidgets.classList.itemSelectionChanged.connect(self.convertDNN)
        self.mmseg_status = False
        self.sam_status = False

        """
        Enhancement Tool
        """
        
        self.use_refinement = False

        """
        Label GrapCut Tool
        """

        # mainWidgets.grabCutButton.clicked.connect(self.checkGrabCutButton)
        # self.use_grabcut = False

        """
        Variables
        """
        self.ControlKey = False
        self.brushSize = 10
        self.EraseSize = 10

        self.input_point_list = []
        self.input_label_list = []

        self.sam_y_idx = []
        self.sam_x_idx = []
    
    def useEnhancement(self, event):
        """
        Label Enhancement Tool
        """
        
        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            if self.x > x :
                min_x = x
                max_x = self.x
            else :
                min_x = self.x
                max_x = x

            if self.y > y :
                min_y = y
                max_y = self.y
            else :
                min_y = self.y
                max_y = y

            # get the region of interest
            img = cvtPixmapToArray(self.pixmap)
            img_roi = img[min_y:max_y, min_x:max_x, :3]
            label_roi = self.label[min_y:max_y, min_x:max_x]
            label_roi = label_roi.astype(np.uint8)

            label_roi = self.applyDenseCRF(img_roi, label_roi)

            self.label[min_y:max_y, min_x:max_x] = label_roi

            # update colormap
            self.updateColorMap()

        self.x = x
        self.y = y
        
    def enhanceLabel(self):
        """
        Label Enhancement Tool
        """
        # get the current image
        img = cvtPixmapToArray(self.pixmap)

        # get binary image of current label
        current_label = self.label == self.brush_class

        # loop through each object in the label
        labeled = skimage.measure.label(current_label)

        for region in skimage.measure.regionprops(labeled):

            # get the bounding box coordinates
            min_y, min_x, max_y, max_x = region.bbox

            # get the region of interest
            img_roi = img[min_y:max_y, min_x:max_x, :3]
            label_roi = current_label[min_y:max_y, min_x:max_x]
            label_roi = label_roi.astype(np.uint8)

            # run the enhancement algorithm
            label_roi = self.applyDenseCRF(img_roi, label_roi)

            # remain only the largest object
            label_roi = skimage.measure.label(label_roi)
            label_roi = label_roi == np.argmax(np.bincount(label_roi.flat)[1:]) + 1

            # smooth the label
            label_roi = skimage.morphology.binary_closing(label_roi, skimage.morphology.square(3))

            # update the label
            self.label[min_y:max_y, min_x:max_x] = label_roi * self.brush_class

        # update colormap
        self.updateColorMap()

    def set_button_state(self, use_autolabel=False, use_refinement=False, use_brush=False, use_erase=False):
        """
        Set the state of the buttons
        """
        self.use_autolabel = use_autolabel
        self.use_refinement = use_refinement
        self.use_brush = use_brush
        self.use_erase = use_erase

        mainWidgets.brushButton.setChecked(use_brush)
        mainWidgets.eraseButton.setChecked(use_erase)
        mainWidgets.autoLabelButton.setChecked(use_autolabel)
        mainWidgets.enhancementButton.setChecked(use_refinement)
        


    def checkAutoLabelButton(self):
        """
        Enable or disable auto label button
        """
        if self.use_autolabel == False:

            self.set_button_state(use_autolabel=True, use_brush=False, use_erase=False)
            
            if hasattr(self, 'EraseMenu'):
                self.EraseMenu.close()  
            if hasattr(self, 'BrushMenu'):
                self.BrushMenu.close()  

            # if self.brush_class == 1:
            #     if hasattr(self, 'mmseg_model') == False :
            #         self.load_mmseg(self.mmseg_config, self.mmseg_checkpoint)
            # else:
            #     if hasattr(self, 'sam_model') == False :
            #         self.load_sam(self.sam_checkpoint) 

            if self.brush_class != 0:
                if hasattr(self, 'sam_model') == False :
                        self.load_sam(self.sam_checkpoint) 


        elif self.use_autolabel == True:
            self.set_button_state()

    def convertDNN (self):
        """
        convert deep learning model among mmseg & sam
        current brush class is the class before the change
        """
        if self.use_autolabel :
            
            if self.brush_class == 1 :
                if hasattr(self, 'sam_model') == False :
                    self.load_sam(self.sam_checkpoint)
    
            else :
                if hasattr(self, 'mmseg_model') == False :
                    self.load_mmseg(self.mmseg_config, self.mmseg_checkpoint)

                
    def openBrushMenu(self):
        """
        Open or Close brush menu
        """
        if self.use_brush == False:
            self.BrushMenu.show()
            if hasattr(self, 'EraseMenu'):
                self.EraseMenu.close()  

            self.set_button_state(use_brush=True, use_erase=False, use_autolabel=False)

        elif self.use_brush == True:
            self.BrushMenu.close()
            self.set_button_state()

    def openEraseMenu(self):
        """
        Open or Close Erase menu
        """
        if self.use_erase == False:
            self.EraseMenu.show()
            if hasattr(self, 'BrushMenu'):
                self.BrushMenu.close()

            self.set_button_state(use_erase=True, use_brush=False, use_autolabel=False)

        elif self.use_erase == True:
            self.EraseMenu.close()
            self.set_button_state()
                
        

    def changeBrushSize(self, value):
        self.brushSize = value
        self.BrushMenu.ui.brushSizeText.setText(str(value))

    def changeEraseSize(self, value):
        self.EraseSize = value
        self.EraseMenu.ui.eraseSizeText.setText(str(value))


    def deleteImage(self, event):
        self.currentIndex = mainWidgets.treeView.currentIndex().data(QFileSystemModel.FilePathRole)
        self.imgFolderPath, _ = os.path.split(self.currentIndex)

        img_path = self.currentIndex
        label_path = self.convertImagePathToLabelPath(img_path)
        os.remove(img_path)
        os.remove(label_path)
        
        mainWidgets.treeView.model().removeRow(mainWidgets.treeView.currentIndex().row(), mainWidgets.treeView.currentIndex().parent())


    @staticmethod
    def convertImagePathToLabelPath(img_path):
        img_label_folder = img_path.replace('/leftImg8bit/', '/gtFine/')
        img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
        return img_label_folder
    
    def addNewImage(self, event):
        
        self.currentIndex = mainWidgets.treeView.currentIndex().data(QFileSystemModel.FilePathRole)
        
        self.imgFolderPath, filename = os.path.split(self.currentIndex)

        if 'leftImg8bit.png' not in filename:
            self.imgFolderPath = self.currentIndex
        
        img_save_folder = self.imgFolderPath
        img_save_folder = img_save_folder.replace( '_leftImg8bit.png', '')  
    
        img_label_folder = img_save_folder.replace('/leftImg8bit/', '/gtFine/')
        img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '')

        readFilePath = QFileDialog.getOpenFileNames(
                caption="Add images to current working directory", filter="Images (*.png *.jpg *.tiff)"
                )
        images = readFilePath[0]

        for img in images:
                
            temp_img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            img_filename = os.path.basename(img) # -> basename is file name
            img_filename = img_filename.replace(' ', '')
            img_filename = img_filename.replace('.jpg', '.png')
            img_filename = img_filename.replace('.JPG', '.png')
            img_filename = img_filename.replace('.tiff', '.png')
            img_filename = img_filename.replace('.png', '_leftImg8bit.png')

            img_gt_filename = img_filename.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
            gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

            _, org_img = cv2.imencode(".png", temp_img)
            org_img.tofile(os.path.join(img_save_folder, img_filename))

            _, gt_img = cv2.imencode(".png", gt)
            gt_img.tofile(os.path.join(img_label_folder, img_gt_filename))

        # self.resetTreeView(refreshIndex=True)
        
    def resetTreeView(self, refreshIndex = False):
        """Reset the tree view
        Args:
            refreshIndex (bool, optional): Refresh the current index. Defaults to False.
        """
        
        mainWidgets.treeView.reset()
        self.fileModel = QFileSystemModel()
        _imgFolderPath = self.imgFolderPath.replace('/train', '')
        _imgFolderPath = _imgFolderPath.replace('/test', '')
        _imgFolderPath = _imgFolderPath.replace('/val', '')

        self.fileModel.setRootPath(_imgFolderPath)
        
        mainWidgets.treeView.setModel(self.fileModel)
        mainWidgets.treeView.setRootIndex(self.fileModel.index(_imgFolderPath))

        if refreshIndex:
            mainWidgets.treeView.setCurrentIndex(self.fileModel.index(self.currentIndex))

    
    def openImage(self, index):
        self.imgPath = self.fileModel.filePath(index)

        if os.path.isdir(self.imgPath):
            print(f"folder")
        
        elif os.path.isfile(self.imgPath):
            
            self.labelPath = self.imgPath.replace('/leftImg8bit/', '/gtFine/')
            self.labelPath = self.labelPath.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
            self.pixmap = readImageToPixmap(self.imgPath)        
            
            self.label = imread(self.labelPath, checkImg=False)
            
            self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.scene = QGraphicsScene()
            self.pixmap_item = self.scene.addPixmap(self.pixmap)

            self.color_pixmap_item = self.scene.addPixmap(self.color_pixmap)

            mainWidgets.mainImageViewer.setScene(self.scene)

            self.scale = mainWidgets.scrollAreaImage.height() / self.label.shape[0]
            mainWidgets.mainImageViewer.setFixedSize(self.scale * self.color_pixmap.size())
            mainWidgets.mainImageViewer.fitInView(self.pixmap_item)

            if hasattr(self, 'sam_predictor'):
                self.set_sam_image()
                self.sam_x_idx = [] 
                self.sam_y_idx = []

        
    def useGrabCut(self, event):
        """
        Label Enhancement Tool
        """
        import cv2
        
        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            if self.x > x :
                min_x = x
                max_x = self.x
            else :
                min_x = self.x
                max_x = x

            if self.y > y :
                min_y = y
                max_y = self.y
            else :
                min_y = self.y
                max_y = y

            # get the region of interest
            img = cvtPixmapToArray(self.pixmap)
            img_roi = img[min_y:max_y, min_x:max_x, :3]
            label_roi = self.label[min_y:max_y, min_x:max_x]
            label_roi = label_roi.astype(np.uint8)

            mask = label_roi == self.brush_class
            
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            mask, bgdModel, fgdModel = cv2.grabCut(img_roi, mask, None, bgdModel, fgdModel,5, cv2.GC_INIT_WITH_MASK)

            mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

            self.label[min_y:max_y, min_x:max_x] = mask * self.brush_class

            # update colormap
            self.updateColorMap()

        self.x = x
        self.y = y

    
    
    
    def drawRectangle(self, event):

        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint(event_global, self.scale)
        
        if (self.fixed_x != x) or (self.fixed_y != y) : 

            # draw empty rectangle on the colormap

            if self.fixed_x > x :
                min_x = x
                max_x = self.fixed_x
            else :
                min_x = self.fixed_x
                max_x = x
            
            if self.fixed_y > y :
                min_y = y
                max_y = self.fixed_y
            else :
                min_y = self.fixed_y
                max_y = y

            # draw rectangle with cv2 

            _colormap = copy.deepcopy(self.colormap)
            
            _colormap = cv2.rectangle(_colormap, (min_x, min_y), (max_x, max_y), (255, 255, 255, 255), 15)

            self.rect_min_x = min_x
            self.rect_max_x = max_x
            self.rect_min_y = min_y
            self.rect_max_y = max_y


            self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
            self.color_pixmap_item.setPixmap(QPixmap())
            self.color_pixmap_item.setPixmap(self.color_pixmap)

        self.x = x
        self.y = y

    def inferenceRectangle(self, event):

        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())
        
        x, y = getScaledPoint(event_global, self.scale)

        if (self.fixed_x != x) or (self.fixed_y != y) :

                
            if x < 0:
                x = 0
            elif x > self.label.shape[1]:
                x = self.label.shape[1]
            
            if y < 0:
                y = 0
            elif y > self.label.shape[0]:
                y = self.label.shape[0]
            
            
            # get the region of interest
            img = cvtPixmapToArray(self.pixmap)
            # cv2.imshow("check_img", img)

            if self.fixed_x > x :
                min_x = x
                max_x = self.fixed_x
            else :
                min_x = self.fixed_x
                max_x = x
            
            if self.fixed_y > y :
                min_y = y
                max_y = self.fixed_y
            else :
                min_y = self.fixed_y
                max_y = y

            self.sam_rec_min_x = min_x
            self.sam_rec_max_x = max_x
            self.sam_rec_min_y = min_y
            self.sam_rec_max_y = max_y

            img_roi = img[min_y:max_y, min_x:max_x, :3]
            
            if self.brush_class == 0:
                pass 

            # elif self.brush_class == 1:
                
            #     mask = self.inference_mmseg(img_roi)

            #     idx = np.argwhere(mask == 1)
            #     y_idx, x_idx = idx[:, 0], idx[:, 1]
            #     x_idx = x_idx + min_x
            #     y_idx = y_idx + min_y

            #     self.label[y_idx, x_idx] = self.brush_class
            #     self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]

            #     _colormap = copy.deepcopy(self.colormap)
            #     cv2.rectangle(_colormap, (min_x, min_y), (max_x, max_y), (255, 255, 255, 255), 3)

            #     self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
            #     self.color_pixmap_item.setPixmap(QPixmap())
            #     self.color_pixmap_item.setPixmap(self.color_pixmap)
                

            else :
                self.input_label_list = []
                self.input_point_list = []

                if (len(self.sam_x_idx) > 0) or (len(self.sam_y_idx) > 0):
                    self.label[self.sam_y_idx, self.sam_x_idx] = self.brush_class
                    self.updateColorMap()

                    self.sam_y_idx = []
                    self.sam_x_idx = []
                
                # dst_bgr = histEqualization_hsv(img_roi)
                img = img[:, :, :3]
                self.sam_predictor.set_image(img)
                

                # save min_x, min_y, max_x, max_y for SAM
                self.sam_min_x = min_x
                self.sam_min_y = min_y
                self.sam_max_x = max_x
                self.sam_max_y = max_y
           
    # def _mouseDoubleClick(self, event):
    #     self.startOrEndSAM()
    
    def _mouseReleasePoint(self, event):

        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())
        x, y = getScaledPoint(event_global, self.scale)

        if self.use_refinement :
            self.useEnhancement(event)

        # if self.use_grabcut : 
        #     self.useGrabCut(event)

        if self.use_autolabel:
            # if mouse is not moved 
            if (x == self.fixed_x) and (y == self.fixed_y) :
                self.inferenceSinglePoint(event)

            else: 
                print("run rectangle inference")
                if (self.rect_max_x - self.rect_min_x) < 100 or (self.rect_max_y - self.rect_min_y) < 100:
                    print("Not Enough")
                    return None
                else:
                    self.inferenceRectangle(event)

        else: 
            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.color_pixmap_item.setPixmap(QPixmap())
            self.color_pixmap_item.setPixmap(self.color_pixmap)
            
    def _mouseMoveEvent(self, event):
        
        if self.use_brush : 
            if self.brushSize > 20: 
                self.useBrush(event)
            else:
                self.useBrushV1(event)

        elif self.use_erase:
            self.useErase(event)

        
        elif self.use_autolabel:
            self.drawRectangle(event)

    def _mousePressPoint(self, event):
        
        """
        Get the brush class and the point where the mouse is pressed
        """

        # Get the brush class
        self.brush_class = mainWidgets.classList.currentRow()
        
        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())
        
        x, y = getScaledPoint(event_global, self.scale)
        
        self.x = x
        self.y = y

        self.fixed_x = x
        self.fixed_y = y 

        # self.input_label_list = []
        # self.input_point_list = []

        # self.input_point_list.append([x, y])
        
        # if self.AltKey == True:
        #     # if mouse left click
        #     if event.button() == Qt.LeftButton:
        #         self.input_label_list.append(1)
                
        #         cv2.circle(self.colormap, (x, y), 30, (255, 0, 0, 255), 3)
        #         cv2.circle(self.colormap, (x, y), 5, (255, 255, 255, 255), -1)

        #     # if mouse right click
        #     elif event.button() == Qt.RightButton:
        #         self.input_label_list.append(0)
                
        #         half_side = 30
        #         vertices = [
        #                         (int(x - half_side), int(y + half_side * np.sqrt(3) / 3)),
        #                         (int(x), int(y - 2 * half_side * np.sqrt(3) / 3)),
        #                         (int(x + half_side), int(y + half_side * np.sqrt(3) / 3))
        #                     ]
                
        #         cv2.circle(self.colormap, (x, y), 5, (255, 255, 255, 255), -1)
        #         cv2.line(self.colormap, vertices[0], vertices[1], (0, 0, 255, 255), 3)
        #         cv2.line(self.colormap, vertices[1], vertices[2], (0, 0, 255, 255), 3)
        #         cv2.line(self.colormap, vertices[2], vertices[0], (0, 0, 255, 255), 3)
            

        #     latest_label = self.input_label_list[-1]
        #     latest_point = self.input_point_list[-1]
        #     sam_ROI = [0, 0, 0, 0]

        #     self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        #     self.color_pixmap_item.setPixmap(QPixmap())
        #     self.color_pixmap_item.setPixmap(self.color_pixmap)

        #     csv_dirname = os.path.dirname(self.labelPath)
        #     csv_dirname = os.path.dirname(csv_dirname)
        #     csv_dirname = os.path.dirname(csv_dirname)
        #     csv_dirname = os.path.join(csv_dirname, "Coordinate")
        #     os.makedirs(csv_dirname, exist_ok=True)
            
        #     csv_filename = os.path.basename(self.labelPath)
        #     csv_filename = csv_filename.replace("_gtFine_labelIds.png", ".csv")
            
        #     csv_path = os.path.join(csv_dirname, csv_filename)
        #     OR_rect = np.zeros(self.label.shape)

        #     getPrompt(sam_ROI, latest_point, latest_label, csv_path, OR_rect)

    def inferenceSinglePoint(self, event):

        # if self.brush_class == 1 : 
        #     img = cvtPixmapToArray(self.pixmap)

        #     min_x = self.x - 128
        #     min_y = self.y - 128
        #     max_x = self.x + 128
        #     max_y = self.y + 128

        #     if min_x < 0 :
        #         min_x = 0
        #     if min_y < 0 :
        #         min_y = 0

        #     if max_x > img.shape[1] :
        #         max_x = img.shape[1]
            
        #     if max_y > img.shape[0] :
        #         max_y = img.shape[0]

        #     img = img[min_y:max_y, min_x:max_x, :]
            
        #     result = self.inference_mmseg(img, do_crf=False)

        #     # update label with result

        #     idx = np.argwhere(result == 1)
        #     y_idx, x_idx = idx[:, 0], idx[:, 1]
        #     x_idx = x_idx + min_x
        #     y_idx = y_idx + min_y

        #     self.label[y_idx, x_idx] = self.brush_class
        #     self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]

        #     _colormap = copy.deepcopy(self.colormap)
        #     cv2.rectangle(_colormap, (min_x, min_y), (max_x, max_y), (255, 255, 255, 255), 3)

        #     self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
        #     self.color_pixmap_item.setPixmap(QPixmap())
        #     self.color_pixmap_item.setPixmap(self.color_pixmap)
        
        # else :
        #     if self.brush_class != 0:
        #         self.inference_sam(event)
        #     elif self.brush_class == 0:
        #         print(f"current brush class is background")

        if self.brush_class != 0:
            self.inference_sam_full(event)
        elif self.brush_class == 0:
            print(f"current brush class is background")

    def inference_sam(self, event):
        """
        Inference the image with the sam model
        Args:
            event (QEvent): The event.
        """
        # cal RoI coords
        sam_x = self.x - self.sam_min_x
        sam_y = self.y - self.sam_min_y

        self.input_point_list.append([sam_x, sam_y])

        # if mouse left click
        if event.button() == Qt.LeftButton:
            self.input_label_list.append(1)
            
            left = True
            right = False          
        
        # if mouse right click
        elif event.button() == Qt.RightButton:
            self.input_label_list.append(0)

            left = False
            right = True
        
        input_point = np.array(self.input_point_list)
        input_label = np.array(self.input_label_list)
        
        if len(self.input_label_list) < 2:

            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            mask = masks[np.argmax(scores), :, :]
            self.sam_mask_input = logits[np.argmax(scores), :, :]

            # update label with result
            idx = np.argwhere(mask == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]

        else : 
            masks, _, _ = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=self.sam_mask_input[None, :, :],
                multimask_output=False,
            )

            mask = masks[0, :, :]
            # self.sam_mask_input = logits

            # update label with result
            idx = np.argwhere(mask == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]

        # cal Full Image coords
        y_idx = y_idx + self.sam_min_y
        x_idx = x_idx + self.sam_min_x

        self.sam_y_idx = y_idx
        self.sam_x_idx = x_idx
        
        self.updateColorMap()

        # cv2 add circles to self.colormap
        # self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]
        
        # self.label[y_idx, x_idx] = self.brush_class
        self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]

        _colormap = copy.deepcopy(self.colormap)
        cv2.rectangle(_colormap, (self.sam_rec_min_x, self.sam_rec_min_y), (self.sam_rec_max_x, self.sam_rec_max_y), (255, 255, 255, 255), 15)

        if left :
            cv2.circle(_colormap, (self.x, self.y), 50, (255, 0, 0, 255), 9)
            cv2.circle(_colormap, (self.x, self.y), 9, (255, 255, 255, 255), -1)
        elif right :
            half_side = 50
            vertices = [
                            (int(self.x - half_side), int(self.y + half_side * np.sqrt(3) / 3)),
                            (int(self.x), int(self.y - 2 * half_side * np.sqrt(3) / 3)),
                            (int(self.x + half_side), int(self.y + half_side * np.sqrt(3) / 3))
                        ]
            
            cv2.circle(_colormap, (self.x, self.y), 9, (255, 255, 255, 255), -1)
            cv2.line(_colormap, vertices[0], vertices[1], (0, 0, 255, 255), 9)
            cv2.line(_colormap, vertices[1], vertices[2], (0, 0, 255, 255), 9)
            cv2.line(_colormap, vertices[2], vertices[0], (0, 0, 255, 255), 9)
        

        
        self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)


        # Save the prompt for comparative experiment
        # sam_ROI = [self.sam_rec_min_x, self.sam_rec_min_y, self.sam_rec_max_x, self.sam_rec_max_y]
        
        # latest_point = self.input_point_list[-1]
        # latest_label = self.input_label_list[-1]

        # csv_dirname = os.path.dirname(self.labelPath)
        # csv_dirname = os.path.dirname(csv_dirname)
        # csv_dirname = os.path.dirname(csv_dirname)
        # csv_dirname = os.path.join(csv_dirname, "Coordinate_SAM")
        # os.makedirs(csv_dirname, exist_ok=True)

        # csv_filename = os.path.basename(self.labelPath)
        # csv_filename = csv_filename.replace("_gtFine_labelIds.png", ".csv")

        # csv_path = os.path.join(csv_dirname, csv_filename)

        # OR_rect = np.zeros(self.label.shape)

        # getPrompt(sam_ROI, latest_point, latest_label, csv_path, OR_rect)

    def inference_sam_full(self, event):
        """
        Inference the full image with the sam model
        Args:
            event (QEvent): The event.
        """
        # # cal RoI coords
        # sam_x = self.x - self.sam_min_x
        # sam_y = self.y - self.sam_min_y

        # Image coords
        sam_x = self.x
        sam_y = self.y


        self.input_point_list.append([sam_x, sam_y])

        # if mouse left click
        if event.button() == Qt.LeftButton:
            self.input_label_list.append(1)
            
            left = True
            right = False          
        
        # if mouse right click
        elif event.button() == Qt.RightButton:
            self.input_label_list.append(0)

            left = False
            right = True
        
        input_point = np.array(self.input_point_list)
        input_label = np.array(self.input_label_list)
        input_box = np.array([self.sam_rec_min_x, self.sam_rec_min_y, self.sam_rec_max_x, self.sam_rec_max_y])
        
        if len(self.input_label_list) < 2:

            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True,
            )

            mask = masks[np.argmax(scores), :, :]
            self.sam_mask_input = logits[np.argmax(scores), :, :] # Choose the model's best mask

            # update label with result
            idx = np.argwhere(mask == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]

        else : 
            masks, _, _ = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=self.sam_mask_input[None, :, :],
                multimask_output=False,
            )

            mask = masks[0, :, :]
            # self.sam_mask_input = logits

            # update label with result
            idx = np.argwhere(mask == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]

        ## cal Full Image coords
        # y_idx = y_idx + self.sam_min_y
        # x_idx = x_idx + self.sam_min_x

        self.sam_y_idx = y_idx
        self.sam_x_idx = x_idx
        
        self.updateColorMap()

        # cv2 add circles to self.colormap
        # self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]
        
        # self.label[y_idx, x_idx] = self.brush_class
        self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]

        _colormap = copy.deepcopy(self.colormap)
        cv2.rectangle(_colormap, (self.sam_rec_min_x, self.sam_rec_min_y), (self.sam_rec_max_x, self.sam_rec_max_y), (255, 255, 255, 255), 15)

        if left :
            cv2.circle(_colormap, (self.x, self.y), 50, (255, 0, 0, 255), 9)
            cv2.circle(_colormap, (self.x, self.y), 9, (255, 255, 255, 255), -1)
        elif right :
            half_side = 50
            vertices = [
                            (int(self.x - half_side), int(self.y + half_side * np.sqrt(3) / 3)),
                            (int(self.x), int(self.y - 2 * half_side * np.sqrt(3) / 3)),
                            (int(self.x + half_side), int(self.y + half_side * np.sqrt(3) / 3))
                        ]
            
            cv2.circle(_colormap, (self.x, self.y), 9, (255, 255, 255, 255), -1)
            cv2.line(_colormap, vertices[0], vertices[1], (0, 0, 255, 255), 9)
            cv2.line(_colormap, vertices[1], vertices[2], (0, 0, 255, 255), 9)
            cv2.line(_colormap, vertices[2], vertices[0], (0, 0, 255, 255), 9)
        

        
        self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)



    def inferenceFullyAutomaticLabeling (self):
        """
        Fully-Automatic Image Labeling 
        """
        damage = [
            "Background",
            "Crack", 
            "Efflorescence", 
            "Rebar-Exposure", 
            "Spalling"
            ]

        # for GUI Image Viewer
        palette = [
            (0, 0, 0, 0),
            (255, 0, 0, 255),  
            (0, 255, 0, 255), 
            (255, 255, 0, 255), 
            (0, 0, 255, 255)
        ]
        # for Saving Processed Image
        palette_rec = [
            (0, 0, 0),
            (0, 0, 255),
            (0, 255, 0), 
            (0, 255, 255), 
            (255, 0, 0)
        ]

        
        damage_name = damage[self.brush_class]
        damage_color = palette[self.brush_class]
        rec_color = palette_rec[self.brush_class]
        
        #######################################
        #      1. First Image Processing      #
        # : Object Detecton (by Faster R-CNN) #
        #######################################

        """
        Load Object Detection model
        """
        # Crack
        if self.brush_class == 1 :
            self.load_mmdet(self.mmdet_crack_config, self.mmdet_crack_checkpoint)
        
        # Efflorescence
        elif self.brush_class == 2 :
            self.load_mmdet(self.mmdet_efflorescence_config, self.mmdet_efflorescence_checkpoint)
        
        # Rebar-Exposure
        elif self.brush_class == 3 :
            self.load_mmdet(self.mmdet_rebarExposure_config, self.mmdet_rebarExposure_checkpoint)
        
        # Spalling
        elif self.brush_class == 4 :
            self.load_mmdet(self.mmdet_spalling_config, self.mmdet_spalling_checkpoint)
        
        """
        Inference Image
        """
        img = cvtPixmapToArray(self.pixmap)
        img = img[:, :, :3]
        bboxes, scores = self.inference_mmdet(img, model=self.mmdet_model)
        thr = 0.6
        box_list = []
        
        for box, score in zip(bboxes, scores):
            if score > thr:
                box_list.append([box, float(score)])
        
        """
        First Image Processing 이미지 저장을 위한 루프
        """
        bbox_colormap = blendImageWithColorMap(img, self.label)
        
        for box in box_list:    
            coord = box[0]
            pred_score = box[1]

            min_x, min_y, max_x, max_y = getScaledPoint_mmdet(coord, scale = 1)
            
            bbox_colormap = cv2.rectangle(bbox_colormap, (min_x, min_y), (max_x, max_y), rec_color, 3)
            
            pred_score = round(float(pred_score), 3) 
            text = f"{damage_name}: {pred_score}"
            text_position = (min_x, min_y-10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = rec_color
            thickness = 1
            line_type = cv2.LINE_AA
            bottomLeftOrigin = False
            cv2.putText(bbox_colormap, text, text_position, font, font_scale, font_color, thickness, line_type, bottomLeftOrigin)
        
        bbox_colormap_path = os.path.dirname(self.labelPath)
        bbox_colormapName = os.path.basename(self.labelPath)
        bbox_colormap_path = os.path.dirname(bbox_colormap_path)
        bbox_colormap_path = os.path.dirname(bbox_colormap_path)
        bbox_colormap_path = os.path.join(bbox_colormap_path, "00. first_imageProcessing_colormap")
        os.makedirs(bbox_colormap_path, exist_ok=True)
        bbox_colormap_path = os.path.join(bbox_colormap_path, bbox_colormapName)

        imwrite_colormap(bbox_colormap_path, bbox_colormap)
            
        #################################################
        #           2. Second Image Processing          #
        # : Semantic Segmentation (by Segment Anything) #
        #################################################
        """
        Load Segment Anything model
        """
        if hasattr(self, 'sam_model') == False :
            self.load_sam(self.sam_checkpoint)
            print(f"load sam")
        
        """
        Inference Image
        """
        self.sam_predictor.set_image(img)
        print(f"set image")    
        for box in box_list :
            coord = box[0]
            pred_score = box[1]

            bbox = np.array([coord[0], coord[1], coord[2], coord[3]])

            masks, scores, logits = self.sam_predictor.predict(
                        box=bbox,
                        multimask_output=True
                    )
            mask = masks[np.argmax(scores), :, :]
            # self.sam_mask_input = logits[np.argmax(scores), :, :]

            # update label with result
            idx = np.argwhere(mask == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]
            
            self.updateColorMap()
            self.label[y_idx, x_idx] = self.brush_class
            self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]
        
        """
        GUI Image viewer를 위한 루프
        """
        self.FAL_colormap = copy.deepcopy(self.colormap)

        for box in box_list :
            coord = box[0]
            pred_score = box[1]
                
            min_x, min_y, max_x, max_y = getScaledPoint_mmdet(coord, scale = 1)
                
            self.FAL_colormap = cv2.rectangle(self.FAL_colormap, (min_x, min_y), (max_x, max_y), damage_color, 3)
                
            pred_score = round(float(pred_score), 3) 
            text = f"{damage_name}: {pred_score}"
            text_position = (min_x, min_y-10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_color = damage_color
            thickness = 2
            line_type = cv2.LINE_AA
            bottomLeftOrigin = False
            cv2.putText(self.FAL_colormap, text, text_position, font, font_scale, font_color, thickness, line_type, bottomLeftOrigin)

        self.color_pixmap = QPixmap(cvtArrayToQImage(self.FAL_colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)
        print(f"complete sam")

        """
        Second Image Processing 이미지 저장을 위한 루프
        """
        sam_colormap = blendImageWithColorMap(img, self.label) 

        for box in box_list :
            coord = box[0]
            pred_score = box[1]
                
            min_x, min_y, max_x, max_y = getScaledPoint_mmdet(coord, scale = 1)
                
            sam_colormap = cv2.rectangle(sam_colormap, (min_x, min_y), (max_x, max_y), rec_color, 3)
                
            pred_score = round(float(pred_score), 3) 
            text = f"{damage_name}: {pred_score}"
            text_position = (min_x, min_y-10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = rec_color
            thickness = 1
            line_type = cv2.LINE_AA
            bottomLeftOrigin = False
            cv2.putText(sam_colormap, text, text_position, font, font_scale, font_color, thickness, line_type, bottomLeftOrigin)

        colormapPath = os.path.dirname(self.labelPath)
        colormapName = os.path.basename(self.labelPath)
        colormapPath = os.path.dirname(colormapPath)
        colormapPath = os.path.dirname(colormapPath)
        colormapPath = os.path.join(colormapPath, "01. second_imageProcessing_colormap")
        os.makedirs(colormapPath, exist_ok=True)
        colormapPath = os.path.join(colormapPath, colormapName)

        imwrite_colormap(colormapPath, sam_colormap)

    def startOrEndSAM(self):

        self.input_label_list = []
        self.input_point_list = []
        
        # if not self.sam_mode:
        self.label[self.sam_y_idx, self.sam_x_idx] = self.brush_class
        self.updateColorMap()

        self.sam_y_idx = []
        self.sam_x_idx = []

    
    def updateColorMap(self):
        """
        Update the color map
        """
        self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
        self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)


    def useBrushV1(self, event):

        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            x_btw, y_btw = getCoordBTWTwoPoints(self.x, self.y, x, y)

            x_btw, y_btw = applyBrushSize(x_btw, y_btw, self.brushSize, self.label.shape[1], self.label.shape[0])

            self.label[y_btw, x_btw] = self.brush_class
            self.colormap[y_btw, x_btw, :3] = self.label_palette[self.brush_class]

            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))

            self.color_pixmap_item.setPixmap(QPixmap())
            self.color_pixmap_item.setPixmap(self.color_pixmap)

        self.x = x
        self.y = y

    def useBrush(self, event):

        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            # find max and min x and y
            label_rgb = cv2.cvtColor(self.label, cv2.COLOR_GRAY2RGB)
            cv2.line(label_rgb, (self.x, self.y), (x, y), (self.brush_class, self.brush_class, self.brush_class), self.brushSize)
            self.label = label_rgb[:, :, 0]

            self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))

            self.color_pixmap_item.setPixmap(QPixmap())
            self.color_pixmap_item.setPixmap(self.color_pixmap)

        self.x = x
        self.y = y

    def useErase(self, event):

        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            # find max and min x and y
            label_rgb = cv2.cvtColor(self.label, cv2.COLOR_GRAY2RGB)
            cv2.line(label_rgb, (self.x, self.y), (x, y), (0, 0, 0), self.EraseSize)
            label_erase = label_rgb[:, :, 0]
            
            if self.brush_class != 0 :
                label_erase_erase = label_erase == self.brush_class
                label_erase_label = self.label == self.brush_class
                self.label[label_erase_erase != label_erase_label] = 0

            
            self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))

            self.color_pixmap_item.setPixmap(QPixmap())
            self.color_pixmap_item.setPixmap(self.color_pixmap)

        self.x = x
        self.y = y