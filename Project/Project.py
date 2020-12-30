from tkinter import *
from tkinter.ttk import *
import tkinter as tk


class LoanPredictor():

	def __init__(self, master): 
		window.title("Bank Loan Predictor")
		window.resizable(False, False)
		window_height = 500
		window_width = 500

		screen_width = window.winfo_screenwidth()
		screen_height = window.winfo_screenheight()

		x_cordinate = int((screen_width/2) - (window_width/2))
		y_cordinate = int((screen_height/2) - (window_height/2))

		window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
		
	def openNewWindow(): 
		# Toplevel object which will 
		# be treated as a new window 
		newWindow = Toplevel(window) 

		# sets the title of the 
		# Toplevel widget 
		newWindow.title("New Window") 

		# sets the geometry of toplevel 
		newWindow.geometry("200x200") 

		# A Label widget to show in toplevel 
		Label(newWindow, text ="This is a new window").pack() 

		label = Label(window, text ="This is the main window") 
		label.pack(pady = 10) 


# Displaying the main window
window = Tk()
mainWindow = LoanPredictor(window);
window.mainloop()