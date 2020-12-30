from tkinter import *
from tkinter.ttk import *
import tkinter as tk


class LoanPredictor():

	def __init__(self, master): 
		# Create Main Display Window
		window.title("Bank Loan Predictor")
		window.resizable(False, False)
		window_height = 500
		window_width = 500

		screen_width = window.winfo_screenwidth()
		screen_height = window.winfo_screenheight()

		x_cordinate = int((screen_width/2) - (window_width/2))
		y_cordinate = int((screen_height/2) - (window_height/2))

		window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
		
		
		# Construct Menubar
		menu = Menu(window)
		
		# Create Label
		canvas = tk.Canvas(window, bg="pink",width="500",height = "30").grid(sticky = W, column=0,row=0) 
		labelMain = tk.Label(window, bg="pink", fg="white", text ="Predictive Model", font=('Helvetica', 15, 'bold'), justify=LEFT).grid(sticky = W, column=0,row=0) 
		self.createSubCategory1()
		self.createSubCategory2()
		self.createSubCategory3()
		self.createSubCategory4()		
	# Create Sub-Categories
	# SubCategory1 = Employment_Type
	def createSubCategory1(self):
		labelSubCat1 = Label(window, text ="Employment Type", font=('Helvetica', 10), justify=LEFT).grid(sticky = W, column=0,row=1)
		
	# SubCategory2 = Credit_Card_types
	def createSubCategory2(self):
		labelSubCat1 = Label(window, text ="Type of Credit Cards", font=('Helvetica', 10), justify=LEFT).grid(sticky = W, column=0,row=2) 	

	# SubCategory3 = Property_Type
	def createSubCategory3(self):
		labelSubCat1 = Label(window, text ="Type of Properties", font=('Helvetica', 10), justify=LEFT).grid(sticky = W, column=0,row=3) 

	# SubCategory4 = Total_Income
	def createSubCategory4(self):
		labelSubCat1 = Label(window, text ="Total Income", font=('Helvetica', 10), justify=LEFT).grid(sticky = W, column=0,row=4) 		

# Displaying the main window
window = Tk()
mainWindow = LoanPredictor(window);
window.mainloop()