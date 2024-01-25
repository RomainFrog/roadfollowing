import tkinter as tk
from PIL import ImageTk, Image
import os
import shutil

class ImageLabeler:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.current_index = 0
        self.labels = []
        self.root = tk.Tk()
        self.label = tk.Label(self.root)
        self.label.pack()
        self.positive_button = tk.Button(self.root, text='Positive (Up Arrow)', command=self.positive_label)
        self.positive_button.pack(side='left')
        self.negative_button = tk.Button(self.root, text='Negative (Down Arrow)', command=self.negative_label)
        self.negative_button.pack(side='left')
        self.quit_button = tk.Button(self.root, text='Quit', command=self.quit)
        self.quit_button.pack(side='right')
        self.count_label = tk.Label(self.root, text=f'{len(self.image_paths)} images left to label')
        self.count_label.pack()
        self.root.bind('<Up>', lambda event: self.positive_label())
        self.root.bind('<Down>', lambda event: self.negative_label())
        self.show_current_image()

    def show_current_image(self):
        # Load the current image
        image_path = self.image_paths[self.current_index]
        image = Image.open(image_path)
        image = image.resize((500, 500), Image.ANTIALIAS)

        # Display the image
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo

    def positive_label(self):
        # Label the current image as positive
        self.labels.append('positive')
        self.move_current_image_to_subdir('positive')
        self.next_image()

    def negative_label(self):
        # Label the current image as negative
        self.labels.append('negative')
        self.move_current_image_to_subdir('negative')
        self.next_image()

    def move_current_image_to_subdir(self, label):
        # Move the current image to a subdirectory based on its label
        image_path = self.image_paths[self.current_index]
        image_name = os.path.basename(image_path)
        subdir_path = os.path.join(os.path.dirname(image_path), label)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        shutil.move(image_path, os.path.join(subdir_path, image_name))

    def next_image(self):
        # Move to the next image
        self.current_index += 1
        self.count_label.config(text=f'{len(self.image_paths) - self.current_index} images left to label')
        if self.current_index < len(self.image_paths):
            self.show_current_image()
        else:
            self.quit()

    def quit(self):
        # Quit the application
        self.root.quit()
        self.root.destroy()


if __name__ == '__main__':
    # Replace this with a list of your image paths
    dir_path = "posneg_dataset"
    files = []
    for file_name in os.listdir(dir_path):
        # Join the directory path and file name to get the full file path
        file_path = os.path.join(dir_path, file_name)
        # Only append the file name if it's a regular file (not a directory)
        if os.path.isfile(file_path):
            files.append(dir_path + "/" + file_name)
    labeler = ImageLabeler(files)
    labeler.root.mainloop()