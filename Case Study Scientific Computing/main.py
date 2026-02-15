import tkinter as tk
import func
import numpy as np 
import matplotlib.pyplot as plt
from tkinter import Label, ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import os

def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    return img


root = tk.Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
ttk.Button(frm, text="Upload Image", command=upload_image).grid(column=2, row=0)



lenna_array = func.convert_png_to_array(img)
lenna_greyscale = func.greyscale(lenna_array, method='nor')
D = func.uniformD(lenna_array, nx=10, my = 10)
lenna_overlay = func.overlay(lenna_greyscale,D)
plt.imshow(lenna_overlay)
plt.show()

root.mainloop()
