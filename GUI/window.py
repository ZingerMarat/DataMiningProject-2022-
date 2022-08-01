import os
import sys
import tkinter
from pathlib import Path
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo, showerror
import pandas as pd

from GUI.main import showResults

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''

numBins = 1
numClusters = 1
numNeighbors = 1
full_path_to_results = Path(__file__).parent / "../Files/results.csv"

# switch state of buttons
def switchState(self, btn):
    if (self.get_column_name()):
        if btn["state"] == "disabled":
            btn["state"] = "normal"
        else:
            btn["state"] = "normal"

# error pop-up window
def Errormessage(msg):
    showerror("Error", msg + " Error")

class Window:
    def __init__(self):

        self.depth_of_tree = 3
        self.root = Tk()
        self.root.title("Data Mining Project ")
        self.root.geometry("800x350+400+200")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.exit())
        self.file_path = None
        self.saved_data = {}
        self.file_path_entry = Entry(self.root)

        self.dataframe = None

        self.classificationColumnEntry = Entry(self.root)
        self.classificationColumn = self.classificationColumnEntry.get()

        self.deleting_rows = StringVar(value="Yes")
        self.completing_missing_values = StringVar(value="Classification value")
        self.normalization = StringVar(value="Yes")
        self.discretization = StringVar(value="Without")
        self.model_selected = StringVar(value="None")

    # create main window
    def draw_widgets(self):
        ### import file
        Label(self.root, text="File Path:", justify=LEFT).grid(row=0, column=0, sticky=W, pady=5, padx=10)
        self.file_path_entry.grid(row=0, column=1, sticky=W + E, padx=10, pady=5)
        Button(self.root, text="Import", command=self.open_file_by_directory).grid(row=0, column=2, sticky=W, pady=5,
                                                                                   padx=10)

        ### Classification Column
        Label(self.root, text="Classification Column:", justify=LEFT).grid(row=1, column=0, sticky=W, pady=5, padx=10)
        self.classificationColumnEntry.grid(row=1, column=1, sticky=W + E, padx=10, pady=5)
        Button(self.root, text="Save Classification Column",
               command=lambda: switchState(self, saveBtn)).grid(row=1, column=2)

        # row 4  Completing missing values:
        Label(self.root, text="Complete missing values:", justify=LEFT).grid(row=4, column=0, sticky=W, pady=5,
                                                                             padx=10)
        Radiobutton(self.root, text="Classification value", variable=self.completing_missing_values,
                    value="Classification value").grid(row=4, column=1, sticky=W, pady=5, padx=10)
        Radiobutton(self.root, text="All data", variable=self.completing_missing_values, value="All data").grid(row=4,
                                                                                                                column=2,
                                                                                                                sticky=W,
                                                                                                                pady=5,
                                                                                                                padx=10)

        # row 5  Normalization
        Label(self.root, text="Normalization:", justify=LEFT).grid(row=5, column=0, sticky=W, pady=5, padx=10)
        Radiobutton(self.root, text="Yes", variable=self.normalization, value="Yes").grid(row=5, column=1, sticky=W,
                                                                                          pady=5, padx=10)
        Radiobutton(self.root, text="No", variable=self.normalization, value="No").grid(row=5, column=2, sticky=W,
                                                                                        pady=5, padx=10)

        # row 6  Discretization
        Label(self.root, text="Discretization:", justify=LEFT).grid(row=6, column=0, sticky=W, pady=5, padx=10)
        Radiobutton(self.root, text="Without", variable=self.discretization,
                    value="Without").grid(row=6, column=1,
                                          sticky=W, pady=5,
                                          padx=10)

        Radiobutton(self.root, text="Equal depth",
                    command=lambda: self.newWindowDiscrete('Number of bins?', range(1, 20, 1)),
                    variable=self.discretization,
                    value="Equal depth").grid(row=6,
                                              column=2,
                                              sticky=W,
                                              pady=5,
                                              padx=10)
        Radiobutton(self.root, text="Equal width",
                    command=lambda: self.newWindowDiscrete('Number of bins?', range(1, 20, 1)),
                    variable=self.discretization,
                    value="Equal width").grid(row=6,
                                              column=3,
                                              sticky=W,
                                              pady=5,
                                              padx=10)
        Radiobutton(self.root, text="Entropy",
                    command=lambda: self.newWindowDiscrete('Number of bins?', range(2, 5, 1)),
                    variable=self.discretization,
                    value="Entropy").grid(row=6, column=4,
                                          sticky=W, pady=5,
                                          padx=10)

        # row 8 The model algorithm
        Label(self.root, text="The model algorithm:", justify=LEFT).grid(row=8, column=0, sticky=W, pady=5, padx=10)
        Radiobutton(self.root, text="Our Decision Tree", variable=self.model_selected, value="OurDT",
                    command=lambda: self.newWindowDepthOfTree('Depth of the tree?', range(1, 51, 1))).grid(row=8,
                                                                                                           column=1,
                                                                                                           sticky=W,
                                                                                                           pady=5,
                                                                                                           padx=10)
        Radiobutton(self.root, text="Sklearn Decision Tree", variable=self.model_selected, value="SklearnDT",
                    command=lambda: self.newWindowDepthOfTree('Depth of the tree?', range(1, 51, 1))).grid(
            row=8, column=2,
            sticky=W, pady=5,
            padx=10)
        Radiobutton(self.root, text="Our Naive Bayes", variable=self.model_selected, value="OurNB").grid(row=9,
                                                                                                         column=1,
                                                                                                         sticky=W,
                                                                                                         pady=5,
                                                                                                         padx=10)
        Radiobutton(self.root, text="Sklearn Naive Bayes", variable=self.model_selected, value="SklearnNB").grid(row=9,
                                                                                                                 column=2,
                                                                                                                 sticky=W,
                                                                                                                 pady=5,
                                                                                                                 padx=10)

        Radiobutton(self.root, text="KNN", variable=self.model_selected, value="KNN",
                    command=lambda: self.newWindowKnn('Number of Neighbors?', range(2, 20, 1))).grid(row=10, column=1,
                                                                                                     sticky=W,
                                                                                                     pady=5, padx=10)
        Radiobutton(self.root, text="K-MEANS", variable=self.model_selected, value="K-MEANS",
                    command=lambda: self.newWindowKmeans('Number of Clusters?', range(2, 20, 1))).grid(row=10, column=2,
                                                                                                       sticky=W, pady=5,
                                                                                                       padx=10)

        # Build & Run button
        saveBtn = Button(self.root, text="Build & Run", width=10,
                         command=lambda: [self.save_data(), switchState(self, runModelBtn)], state=DISABLED)
        saveBtn.grid(row=11, column=2, pady=10, sticky=E)

        # Models result button
        runModelBtn = Button(self.root, text="Models result", width=15, command = lambda: showResults(), state=DISABLED)
        runModelBtn.grid(row=11, column=3, pady=10)

        # Open all Results button
        Button(self.root, text="Open all Results", width=15, command=lambda: os.system(str(full_path_to_results))).grid(row=11, column=4, pady=10, sticky=W)

        # Exit button
        Button(self.root, text="Exit", width=10, command=self.exit).grid(row=11, column=5, pady=10, sticky=E)

    # open file
    def open_file_by_directory(self):
        self.file_path = filedialog.askopenfilename(initialdir=self.file_path_entry.get(),
                                                    filetypes=[("CSV files", ".csv")])
        try:
            path = os.path.isfile(self.file_path)
            if path is True:
                self.dataframe = pd.read_csv(self.file_path)
                if self.dataframe.empty:
                    Errormessage("File is Empty")
                else:
                    showinfo(title='OK', message="File Selected")
                    self.file_path_entry.insert(END, self.file_path)
        except:
            Errormessage("File")

    # get classifier column name
    def get_column_name(self):
        try:
            if self.classificationColumnEntry.get() in self.dataframe.columns:
                self.classificationColumn = self.classificationColumnEntry.get()
                return True
            else:
                Errormessage("classification column")
                return False
        except AttributeError:
            Errormessage("'NoneType' object has no attribute 'columns'")

    # new window for depth selection
    def newWindowDepthOfTree(self, question, range1):
        def choose(window):
            global depth
            depth = slider1.get()
            self.depth_of_tree = depth

            window.quit()
            window.destroy()

        window = Toplevel(self.root)
        window.geometry("300x200+700+300")
        window.title("Settings")

        Label(window, text='').pack()
        Label(window, text=question).pack()
        lst = [i for i in range1]
        slider1 = Scale(window, from_=min(lst), to=max(lst), orient="horizontal")
        slider1.pack()

        Button(window, text="Continue", padx=25, pady=10, command=lambda: choose(window)).pack()
        window.mainloop()

    # new window for num of bins selection
    def newWindowDiscrete(self, question, range1):
        def choose(window):
            global numBins
            numBins = 1

            numBins = slider1.get()

            window.quit()
            window.destroy()

        window = tkinter.Toplevel(self.root)
        window.geometry("300x200")
        window.title("Settings")

        tkinter.Label(window, text='').pack()
        tkinter.Label(window, text=question).pack()
        lst = [i for i in range1]
        slider1 = tkinter.Scale(window, from_=min(lst), to=max(lst), orient="horizontal")
        slider1.pack()

        tkinter.Button(window, text="Continue", padx=25, pady=10, command=lambda: choose(window)).pack()
        window.mainloop()

    # new window for num of neighbors selection
    def newWindowKnn(self, question, range1):
        def choose(window):
            global numNeighbors
            numNeighbors = 1

            numNeighbors = slider1.get()

            window.quit()
            window.destroy()

        window = tkinter.Toplevel(self.root)
        window.geometry("300x200")
        window.title("Settings")

        tkinter.Label(window, text='').pack()
        tkinter.Label(window, text=question).pack()
        lst = [i for i in range1]
        slider1 = tkinter.Scale(window, from_=min(lst), to=max(lst), orient="horizontal")
        slider1.pack()

        tkinter.Button(window, text="Continue", padx=25, pady=10, command=lambda: choose(window)).pack()
        window.mainloop()

    # new window for num of clusters selection
    def newWindowKmeans(self, question, range1):
        def choose(window):
            global numClusters
            numClusters = 1

            numClusters = slider1.get()

            window.quit()
            window.destroy()

        window = tkinter.Toplevel(self.root)
        window.geometry("300x200")
        window.title("Settings")

        tkinter.Label(window, text='').pack()
        tkinter.Label(window, text=question).pack()
        lst = [i for i in range1]
        slider1 = tkinter.Scale(window, from_=min(lst), to=max(lst), orient="horizontal")
        slider1.pack()

        tkinter.Button(window, text="Continue", padx=25, pady=10, command=lambda: choose(window)).pack()
        window.mainloop()

    # save all user's choises
    def save_data(self):
        from GUI import main
        self.saved_data = {"URL": self.file_path,
                           "dataframe": self.dataframe,
                           "classification_column": self.classificationColumn,
                           "deleting_rows": self.deleting_rows.get(),
                           "completing_missing_values": self.completing_missing_values.get(),
                           "normalization": self.normalization.get(),
                           "discretization": self.discretization.get(),
                           "num_of_bins": numBins,
                           "num_of_neighbors": numNeighbors,
                           "num_of_clusters": numClusters,
                           "depth_of_tree": self.depth_of_tree,
                           "model_selected": self.model_selected.get()
                           }

        main.choices = self.saved_data
        main.splitTheData()
        main.runTheModels()

    # end of the program
    def exit(self):
        self.root.destroy()
        sys.exit()

    # run method
    def run(self):
        self.draw_widgets()
        self.root.mainloop()
