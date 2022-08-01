import pickle
import tkinter as tk
from pathlib import Path
from tkinter.messagebox import showerror, showinfo

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''


# create and save model
def createPickle(model):
    """
    this function get model as parameter and creates pickle of it.
    :return: none
    """
    try:
        path_model = Path(__file__).parent / "../models_pickle/Model"
        pickle.dump(model, open(path_model, 'wb'))
        showinfo("Success", "File named 'Model' created")
    except:
        showerror("Error", "Failed to create pickle")


# print plot function
def printPlot(results):
    # create a new window
    win = tk.Tk()
    win.geometry("1500x600+10+100")
    win.title("Model's Score")

    data = {'values': ['Accuracy', 'Precision', 'Recall', 'Majority', 'F1-Score'],
            'percentage': [results['accuracy'], results['precision'], results['recall'], results['majority'],
                           results['f1Score']]}

    df = DataFrame(data, columns=['values', 'percentage'])

    # create bar plot
    figure = plt.Figure(figsize=(6.5, 4.5), dpi=100)
    ax = figure.add_subplot()
    bar = FigureCanvasTkAgg(figure, win)
    bar.get_tk_widget().grid(column=1, row=0)
    df = df[['values', 'percentage']].groupby('values').sum()
    df.plot(kind='bar', legend=False, ax=ax, rot=0, xlabel='', ylabel='percentage')
    ax.set_title(
        'Accuracy: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%,\nMajority: {:.3f}%, F1-Score: {:.3f}%'.format(
            float(results['accuracy']),
            float(results['precision']),
            float(results['recall']),
            float(results['majority']),
            float(results['f1Score'])))

    # create confusion matrix for test file
    conf_matrix = results['confusion_matrix']
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.matshow(conf_matrix, cmap="BuPu", alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='medium')

    plt.xlabel('Predicted Values', fontsize=10)
    plt.ylabel('Actual Values', fontsize=10)
    plt.title('Test Confusion Matrix', fontsize=14)

    bar = FigureCanvasTkAgg(fig, win)
    bar.get_tk_widget().grid(column=2, row=0)
    plt.show()

    # confusion matrix for train file
    conf_matrix = results['confusion_matrix_train']
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.matshow(conf_matrix, cmap="BuPu", alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='medium')

    plt.xlabel('Predicted Values', fontsize=10)
    plt.ylabel('Actual Values', fontsize=10)
    plt.title('Train Confusion Matrix', fontsize=14)

    bar = FigureCanvasTkAgg(fig, win)
    bar.get_tk_widget().grid(column=3, row=0)
    plt.show()

    # create model button
    tk.Button(win, text='Create Model', font=('serif', 10, 'bold'),
              command=lambda: createPickle(results['model']), padx=25,
              pady=10).grid(column=2, row=2)

    win.configure(background='white')
    win.mainloop()
