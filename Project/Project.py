from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter as tk
import pandas as pd

class LoanPredictor():

	def __init__(self): 
		# Create Main Display Window
		window = tk.Tk()
		self.dtFrame = Frame()
		self.knnFrame = Frame()
		self.nbFrame = Frame()
		self.svmFrame = Frame()
		self.mlt1Selected = ''
		self.mlt2Selected = ''
		self.mlt3Selected = ''
		self.mlt4Selected = ''
		self.employmentType = ''
		self.propertyType =''
		self.cardType = ''
		self.mthSalary = ''
		self.loanAmount = ''
		self.window = window
		self.window.title("Bank Loan Predictor")
		self.window.resizable(False, False)
		window_height = 600
		window_width = 600

		screen_width = self.window.winfo_screenwidth()
		screen_height = self.window.winfo_screenheight()

		x_cordinate = int((screen_width/2) - (window_width/2))
		y_cordinate = int((screen_height/3.3) - (window_height/3.3))

		self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
		
		# Construct Menubar
		menubar = tk.Menu(self.window)
		self.window.config(menu=menubar)
		
		technicMenu = tk.Menu(menubar, tearoff = 0)
		self.mlt1Selected = BooleanVar()		
		self.mlt2Selected = BooleanVar()		
		self.mlt3Selected = BooleanVar()		
		self.mlt4Selected = BooleanVar()
		technicMenu.add_radiobutton(label = "Decision Tree Classifier", variable = self.mlt1Selected, value=True, command = self.runDT)	
		technicMenu.add_radiobutton(label = "K-Nearest Neighbour", variable = self.mlt2Selected, value=True, command = self.runKNN)
		technicMenu.add_radiobutton(label = "Naive Bayes", variable = self.mlt3Selected, value=True, command = self.runNB)
		technicMenu.add_radiobutton(label = "SVM Kernel", variable = self.mlt4Selected, value=True, command = self.runSVM)
		menubar.add_cascade(label = "Machine Learning Technique(s)", menu = technicMenu)	
		
		helpMenu = tk.Menu(menubar, tearoff = 0)
		helpMenu.add_command(label = "About Us", command = self.aboutUs)	
		helpMenu.add_command(label = "How to Use")			
		menubar.add_cascade(label = "Help", menu = helpMenu)
		
		self.mainDisplay()

	
	# Create Main Display
	def mainDisplay(self):
		canvas = tk.Canvas(self.window, bg="pink",width="600",height = "40").place(x=0,y=0)
		labelMain = tk.Label(self.window, bg="pink", fg="white", text ="Prediction Model", font=('Helvetica', 15, 'bold')).place(x=220,y=5)
		labelWarning = tk.Label(self.window, text ="Please Select a Machine Learning Technique in order to predict.", font=('Helvetica', 12)).place(x=70,y=370)
		self.createSubCategory1()
		self.createSubCategory2()
		self.createSubCategory3()
		self.createSubCategory4()
		self.createSubCategory5()
		resetBtn = Button(self.window, text ="Reset", command=self.resetButtonOnClicked).place(x=200,y=500, height=30, width=100) 	
		predictBtn = Button(self.window, text ="Predict Now", command=self.predictionButtonOnClicked).place(x=300,y=500, height=30, width=100)
		
	# Machine Learning Technique(s) Menu
	# Decision Tree Classifier On Selected in Menu
	def runDT(self):
		self.checkMLT()
		self.mlt1Selected.set(True)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.dtFrame = tk.Frame(self.window, width=600, height=50)
		self.dtFrame.place(x=210,y=400)
		dtButton = Button(self.dtFrame, text ="Generate Decision Tree Classifier").grid(row=1, column=0)
	# K-Nearest Neighbour On Selected in Menu	
	def runKNN(self):
		self.checkMLT()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(True)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.knnFrame = tk.Frame(self.window, width=600, height=50)
		self.knnFrame.place(x=210,y=400)
		knnSelector = Scale(self.knnFrame, from_=0, to=9, orient=HORIZONTAL).grid(row=0, column=0)
		knnButton = Button(self.knnFrame, text ="Generate K-Nearest Neighbour").grid(row=1, column=0)
	# Naive Bayes On Selected in Menu
	def runNB(self):
		self.checkMLT()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(True)
		self.mlt4Selected.set(False)	
		self.nbFrame = tk.Frame(self.window, width=600, height=50)
		self.nbFrame.place(x=210,y=400)
		nbButton = Button(self.nbFrame, text ="Generate Naive Bayes").grid(row=1, column=0)
	# SVM On Selected in Menu
	def runSVM(self):
		self.checkMLT()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(True)	
		self.svmFrame = tk.Frame(self.window, width=600, height=50)
		self.svmFrame.place(x=210,y=400)
		var = StringVar()
		kernel1 = Radiobutton(self.svmFrame, text="Rbf", variable=var, value=1,command=self.SVMRbf).grid(row=0, column=0)
		kernel2 = Radiobutton(self.svmFrame, text="Linear", variable=var, value=2,command=self.SVMLinear).grid(row=1, column=0)
		kernel3 = Radiobutton(self.svmFrame, text="Polynomial", variable=var, value=3, command=self.SVMPoly).grid(row=2, column=0)
		svmButton = Button(self.svmFrame, text ="Generate SVM").grid(row=3, column=0)
	# Check if the MLT is Clicked/Selected
	def checkMLT(self):
		if self.mlt1Selected == True:
			for widget in self.knnFrame.winfo_children():
			   widget.destroy()
			self.knnFrame.pack_forget()   
			for widget in self.nbFrame.winfo_children():
			   widget.destroy()
			self.nbFrame.pack_forget() 
			for widget in self.svmFrame.winfo_children():
			   widget.destroy()
			self.svmFrame.pack_forget() 
		if self.mlt2Selected == True:
			for widget in self.dtFrame.winfo_children():
			   widget.destroy()
			self.dtFrame.pack_forget()   
			for widget in self.nbFrame.winfo_children():
			   widget.destroy()
			self.nbFrame.pack_forget() 
			for widget in self.svmFrame.winfo_children():
			   widget.destroy()
			self.svmFrame.pack_forget() 
		if self.mlt3Selected == True:
			for widget in self.dtFrame.winfo_children():
			   widget.destroy()
			self.dtFrame.pack_forget()   
			for widget in self.knnFrame.winfo_children():
			   widget.destroy()
			self.knnFrame.pack_forget() 
			for widget in self.svmFrame.winfo_children():
			   widget.destroy()
			self.svmFrame.pack_forget() 
		if self.mlt4Selected == True:
			for widget in self.dtFrame.winfo_children():
			   widget.destroy()
			self.dtFrame.pack_forget()   
			for widget in self.knnFrame.winfo_children():
			   widget.destroy()
			self.knnFrame.pack_forget() 
			for widget in self.nbFrame.winfo_children():
			   widget.destroy()
			self.nbFrame.pack_forget() 	
		else:
			for widget in self.dtFrame.winfo_children():
			   widget.destroy()
			self.dtFrame.pack_forget()   
			for widget in self.knnFrame.winfo_children():
			   widget.destroy()
			self.knnFrame.pack_forget() 
			for widget in self.nbFrame.winfo_children():
			   widget.destroy()
			self.nbFrame.pack_forget() 		
			for widget in self.svmFrame.winfo_children():
			   widget.destroy()
			self.svmFrame.pack_forget() 			
	# Help Menu
	def aboutUs(self):
		messagebox.showinfo("About Us","TDS3301 - Data Mining Project\nGroup Member:\nOng Shuoh Chwen 1171102212\nYong Wen Kai 1171101664\nLecturer:\nDr. Ting Choo Yee")
	
	def howToUse(self):
		pass
	# Create Sub-Categories
	# SubCategory1 = Employment_Type
	def createSubCategory1(self):
		labelSubCat1 = Label(self.window, text ="Employment Type", font=('Helvetica', 10), justify=LEFT).place(x=10,y=50)
		self.employmentType = StringVar()
		empType1 = Radiobutton(self.window, text="Employee", variable=self.employmentType, value="Employee",command=self.saveSelectedValues).place(x=10,y=70)
		empType2 = Radiobutton(self.window, text="Employer", variable=self.employmentType, value="Employer",command=self.saveSelectedValues).place(x=10,y=90)
		empType3 = Radiobutton(self.window, text="Fresh Graduate", variable=self.employmentType, value="Fresh Graduate", command=self.saveSelectedValues).place(x=10,y=110)
		empType4 = Radiobutton(self.window, text="Self Employment", variable=self.employmentType, value="Self Employment", command=self.saveSelectedValues).place(x=10,y=130)			
		
	# SubCategory2 = Credit_Card_types
	def createSubCategory2(self):
		labelSubCat2 = Label(self.window, text ="Type of Credit Cards", font=('Helvetica', 10), justify=LEFT).place(x=10,y=170)
		self.cardType = StringVar()
		cardType1 = Radiobutton(self.window, text="Normal", variable=self.cardType, value="Normal",command=self.saveSelectedValues).place(x=10,y=190)
		cardType2 = Radiobutton(self.window, text="Gold", variable=self.cardType, value="Gold",command=self.saveSelectedValues).place(x=10,y=210)
		cardType3 = Radiobutton(self.window, text="Platinum", variable=self.cardType, value="Platinum", command=self.saveSelectedValues).place(x=10,y=230)			
		
	# SubCategory3 = Property_Type
	def createSubCategory3(self):
		labelSubCat3 = Label(self.window, text ="Type of Properties", font=('Helvetica', 10), justify=LEFT).place(x=10,y=270) 
		self.propertyType = StringVar()
		propertyType1 = Radiobutton(self.window, text="Bungalow", variable=self.propertyType, value="Bungalow", command=self.saveSelectedValues).place(x=10,y=290)		
		propertyType2 = Radiobutton(self.window, text="Condominium", variable=self.propertyType, value="Condominium",command=self.saveSelectedValues).place(x=10,y=310)
		propertyType3 = Radiobutton(self.window, text="Flat", variable=self.propertyType, value="Flat",command=self.saveSelectedValues).place(x=10,y=330)
		propertyType4 = Radiobutton(self.window, text="Terrace", variable=self.propertyType, value="Terrace",command=self.saveSelectedValues).place(x=10,y=350)
		
	# SubCategory4 = Loan_Amount
	def createSubCategory4(self):
		labelSubCat4 = Label(self.window, text ="Loan Amount (RM)", font=('Helvetica', 10), justify=LEFT).place(x=350,y=50)
		self.loanAmount = StringVar()
		loanAmount1 = Radiobutton(self.window, text="100,000 - 300,000", variable=self.loanAmount, value="100,000 - 300,000", command=self.saveSelectedValues).place(x=350,y=70)
		loanAmount2 = Radiobutton(self.window, text="300,000 - 500,000", variable=self.loanAmount, value="300,000 - 500,000", command=self.saveSelectedValues).place(x=350,y=90)
		loanAmount3 = Radiobutton(self.window, text="500,000 - 700,000", variable=self.loanAmount, value="500,000 - 700,000", command=self.saveSelectedValues).place(x=350,y=110)
		loanAmount4 = Radiobutton(self.window, text="700,000 - 900,000", variable=self.loanAmount, value="700,000 - 900,000", command=self.saveSelectedValues).place(x=350,y=130)			

	# SubCategory5 = Monthly_Salary
	def createSubCategory5(self):
		labelSubCat5 = Label(self.window, text ="Monthly Salary (RM)", font=('Helvetica', 10), justify=LEFT).place(x=350,y=170)
		self.mthSalary = StringVar()
		mthSalary1 = Radiobutton(self.window, text="<4,000", variable=self.mthSalary, value="<4,000", command=self.saveSelectedValues).place(x=350,y=190)
		mthSalary2 = Radiobutton(self.window, text="4,000 - 7,000", variable=self.mthSalary, value="4,000 - 7,000", command=self.saveSelectedValues).place(x=350,y=210)
		mthSalary3 = Radiobutton(self.window, text="7,000 - 10,000", variable=self.mthSalary, value="7,000 - 10,000", command=self.saveSelectedValues).place(x=350,y=230)
		mthSalary4 = Radiobutton(self.window, text="10,000 - 13,000", variable=self.mthSalary, value="10,000 - 13,000", command=self.saveSelectedValues).place(x=350,y=250)	

	# Reset Button On Clicked
	def resetButtonOnClicked(self):
		self.mainDisplay()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.checkMLT()
		
	# Prediction Button On Clicked
	def predictionButtonOnClicked(self):
		pass
	
	def saveSelectedValues(self):
		print (self.employmentType.get() + "\n" + 
			   self.cardType.get() + "\n" +
			   self.propertyType.get() + "\n" +		
			   self.loanAmount.get() + "\n" +
			   self.mthSalary.get() + "\n")
		#employmentType = 
	# Generate SVM Accordingly
	def SVMRbf(self): 
		pass
	def SVMLinear(self): 
		pass
	def SVMPoly(self): 
		pass
# Displaying the main window

mainWindow = LoanPredictor()
mainWindow.window.mainloop()