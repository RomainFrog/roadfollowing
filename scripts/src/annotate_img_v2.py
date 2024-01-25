#!/usr/bin/env python
# coding: utf-8


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# /!\ HOW TO LAUNCH THE SCRIPT /!\
#      In a terminal in the directory of the script write this in the command prompt : python annotate_img.py input_folder output_folder
#     
#      Note : Check that you have installed all the required packages
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk, ImageDraw
import subprocess
import sys
import uuid


class ImageEditor:
    def __init__(self, master, input_folder, output_folder, category):
        self.master = master
        #self.images = []
        self.current_image_index = 0
        self.pixel_pos = None
        self.output_folder = output_folder + "/" + category
        self.input_folder = input_folder
        self.img_name = []
        self.img_realname = ''
        self.compt = 0

        # Création du dossier d'annotations
        os.makedirs(self.output_folder, exist_ok=True)

        #Chargement des images à annoter
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, filename)
                self.img_name.append(img_path)
                #img = Image.open(img_path)
                #img = self.resize_image(img)
                #self.images.append(img)
        

        # Création des widgets
        self.canvas = tk.Canvas(self.master, width=1000, height=1000)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.pack(side=tk.TOP)
        
        self.skip_btn = tk.Button(self.master, text="Skip Image", command=self.skip_current_image)
        self.skip_btn.pack(side=tk.LEFT)
        
        self.show_current_image()
        
    def resize_image(self, img):
        max_size = 1000000
        img.thumbnail((max_size, max_size), Image.ANTIALIAS)
        return img

    # def show_current_image(self):
        # Affichage de l'image courante
    #     self.image = self.resize_image(Image.open(self.img_name[self.current_image_index]))
    #     self.photo = ImageTk.PhotoImage(self.image)
    #    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
    def show_current_image(self):
        # Affichage de l'image courante
        self.image = self.resize_image(Image.open(self.img_name[self.current_image_index]))
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        width, height = self.image.size
        self.canvas.create_line(0, height/2, width, height/2, fill='red', width=2)


    def on_click(self, event):
        # Récupération des coordonnées du pixel cliqué
        self.pixel_pos = (event.y, event.x)
        print(f"Coordonnées du pixel cliqué : {self.pixel_pos}, Image : {self.compt}")
        self.compt += 1

        # Enregistrement de l'image avec les coordonnées dans le nom de fichier
        if self.image and self.pixel_pos:
            filename = '%d_%d_%s.jpg' % (self.pixel_pos[0], self.pixel_pos[1], str(uuid.uuid1()))
            self.image.save(os.path.join(self.output_folder, filename))

        # Passage à l'image suivante
        if self.current_image_index < len(self.img_name) - 1:
            self.current_image_index += 1
            self.show_current_image()
        else:
            # Toutes les images ont été annotées
            self.show_end_message()
            
    def skip_current_image(self):
    # Passage à l'image suivante sans annoter l'image actuelle
        if self.current_image_index < len(self.img_name) - 1:
            self.current_image_index += 1
            self.show_current_image()
            print("Refus d'annoter l'image, passage à l'image suivante")
        else:
            # Toutes les images ont été annotées
            self.show_end_message()
            
    def show_end_message(self):
        tk.messagebox.showinfo("Done!", "All images have been annotated.")



input_folder, task, category = sys.argv[1:4]

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root, input_folder, task, category)
    root.mainloop()

