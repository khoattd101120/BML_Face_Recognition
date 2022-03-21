import sys

sys.path.append(r'..\src')
sys.path.append(r'..\insightface\deploy')
sys.path.append(r'..\insightface\common')

from CreateClassifier import CreateClassifier

import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage
from Detector import main_app
from CaptureFace import *
from GUI_TrainFeature import *

names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.configure(bg="#ffffff")
        global names
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("VinBDI Face Recognition")
        self.resizable(False, False)
        self.geometry("500x250")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.recognizer = CreateClassifier(embedding_path='../src/outputs/testmodelv1.pickle')
        self.active_name = None
        self.num_of_images = 0
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageThree, PageFour):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            f = open("nameslist.txt", "a+")
            for i in names:
                f.write(i + " ")
            self.destroy()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg="#ffffff")
        # load = Image.open("homepagepic.png")
        # load = load.resize((250, 250), Image.ANTIALIAS)
        render = PhotoImage(file='homepagepic.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=1, rowspan=4, sticky="nsew")
        label = tk.Label(self, text="        Home Page        ", font=self.controller.title_font, fg="#263942",
                         bg="#ffffff")
        label.grid(row=0, sticky="ew")
        button1 = tk.Button(self, text="   Add a User  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="   Check a User  ", fg="#ffffff", bg="#263942", command=self.openwebcam
                            # command=lambda: self.controller.show_frame("PageTwo")
                            )
        button3 = tk.Button(self, text="Quit", fg="#263942", bg="#ffffff", command=self.on_closing)
        button1.grid(row=1, column=0, ipady=3, ipadx=7)
        button2.grid(row=2, column=0, ipady=3, ipadx=2)
        button3.grid(row=3, column=0, ipady=3, ipadx=32)

    def openwebcam(self):
        main_app(self.controller.active_name, self.controller.recognizer)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            with open("nameslist.txt", "w") as f:
                for i in names:
                    f.write(i + " ")
            self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg="#ffffff")
        tk.Label(self, text="Enter the name", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=10,
                                                                                           padx=5)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttonext = tk.Button(self, text="Next", fg="#ffffff", bg="#263942", command=self.start_training)
        self.buttoncanc.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)

    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.user_name.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif len(self.user_name.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        name = self.user_name.get()
        names.add(name)
        self.controller.active_name = name
        # self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg="#ffffff")
        self.numimglabel = tk.Label(self, text="Number of images captured = 0", font='Helvetica 12 bold', fg="#263942")
        self.numimglabel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        self.capturebutton = tk.Button(self, text="Capture Data Set", fg="#ffffff", bg="#263942", command=self.capimg)
        self.trainbutton = tk.Button(self, text="Train The Model", fg="#ffffff", bg="#263942", command=self.trainmodel)
        self.capturebutton.grid(row=1, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.trainbutton.grid(row=1, column=1, ipadx=5, ipady=4, padx=10, pady=20)

    def capimg(self):
        self.numimglabel.config(text=str("Captured Images = 0 "))
        messagebox.showinfo("INSTRUCTIONS", "We will Capture 10 pic of your Face.")
        x = start_capture(self.controller.active_name, self.controller.recognizer)
        self.controller.num_of_images = x
        self.numimglabel.config(text=str("Number of images captured = " + str(x)))

    def trainmodel(self):
        # if self.controller.num_of_images < 10:
        #     messagebox.showerror("ERROR", "No enough Data, Capture at least 300 images!")
        #     return
        train_classifer(self.controller.active_name, self.controller.recognizer)
        messagebox.showinfo("SUCCESS", "The model has been successfully trained!")
        self.controller.show_frame("PageFour")


class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg="#ffffff")

        label = tk.Label(self, text="Face Recognition", font='Helvetica 16 bold')
        label.grid(row=0, column=0, sticky="ew")
        button1 = tk.Button(self, text="Face Recognition", command=self.openwebcam, fg="#ffffff", bg="#263942")
        # button2 = tk.Button(self, text="Emotion Detection", command=self.emot, fg="#ffffff", bg="#263942") button3
        # = tk.Button(self, text="Gender and Age Prediction", command=self.gender_age_pred, fg="#ffffff", bg="#263942")
        button4 = tk.Button(self, text="Go to Home Page", command=lambda: self.controller.show_frame("StartPage"),
                            bg="#ffffff", fg="#263942")
        button1.grid(row=1, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        # button2.grid(row=1,column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        # button3.grid(row=2,column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button4.grid(row=1, column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)

    def openwebcam(self):
        main_app(self.controller.active_name, self.controller.recognizer)
    # def gender_age_pred(self):
    #  ageAndgender()
    # def emot(self):
    #   emotion()


if __name__ == '__main__':
    app = MainUI()
    app.iconphoto(False, tk.PhotoImage(file='icon.png'))
    app.mainloop()
