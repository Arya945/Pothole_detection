from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.uix.dropdown import *
from detector import detector
import cv2
import os

cfg_path = r"files\pothole.cfg"
weights_path = r"files\yolov4.weights"
names_path = r"files\data.names"

pothole_detector = detector(cfg_path,weights_path,names_path)

class FileChooserDemo(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        
        self.cwd = os.getcwd()

        # FileChooser widget
        self.filechooser = FileChooserListView(path='.')
        self.img = Image(source = "R.jpeg")
        self.add_widget(self.filechooser)
        self.add_widget(self.img)
        
        # Button to print selected file
        self.button = Button(text='Detect')
        self.button.bind(on_press=self.Detect)
        self.add_widget(self.button)
        
    def Detect(self, instance):
        selection = self.filechooser.selection
        if selection:
            pothole_detector.detect(selection[0], 0.3)
            new = f"{self.cwd}\\output.jpg"
            self.img.source = new

class MyApp(App):
    def build(self):
        return FileChooserDemo()

if __name__ == '__main__':
    MyApp().run()