from tkinter import *
from tkinter import ttk
from tkinter import filedialog

root = Tk()
root.title("decodeGUI")
root.geometry("800x600")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


inputFile = StringVar()
def loadFile():
    fd = filedialog.askopenfile(initialdir="C:/Users/Jonas/projects/personal/matura/py/", title="Select WAV file", filetypes=[("WAV files", ".wav")])
    inputFile.set(fd.name)
    return

ttk.Button(mainframe, text="Load File", command=loadFile).grid(column=0, row=0, sticky=W)
ttk.Label(mainframe, textvariable=inputFile).grid(column=1, row=0, sticky=W)

root.mainloop()