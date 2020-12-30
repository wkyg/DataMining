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
		menubar = tk.Menu(window)
		window.config(menu=menubar)
		
		technicMenu = tk.Menu(menubar, tearoff = 0)
		technicMenu.add_command(label = "Decision Tree Classifier", command = self.runDT())	
		technicMenu.add_command(label = "K-Nearest Neighbour", command = self.runKNN())
		technicMenu.add_command(label = "Naive Bayes", command = self.runNB())
		technicMenu.add_command(label = "SVM Kernel", command = self.runSVM())
		menubar.add_cascade(label = "Machine Learning Technique(s)", menu = technicMenu)	
		
		helpMenu = tk.Menu(menubar, tearoff = 0)
		helpMenu.add_command(label = "About Us")	
		helpMenu.add_command(label = "How to Use")			
		menubar.add_cascade(label = "Help", menu = helpMenu)	
		
		# Create Label
		canvas = tk.Canvas(window, bg="pink",width="500",height = "40").place(x=0,y=0)
		labelMain = tk.Label(window, bg="pink", fg="white", text ="Predictive Model", font=('Helvetica', 15, 'bold')).place(x=5,y=5)
		self.createSubCategory1()
		self.createSubCategory2()
		self.createSubCategory3()
		self.createSubCategory4()
		self.createSubCategory5()			
		self.predictionButton()
		
	# Create Sub-Categories
	# SubCategory1 = Employment_Type
	def createSubCategory1(self):
		labelSubCat1 = Label(window, text ="Employment Type", font=('Helvetica', 10), justify=LEFT).place(x=5,y=50)
		var = IntVar()
		empType1 = Radiobutton(window, text="Employee", variable=var, value=1,command=self.createSubCategory1).place(x=5,y=70)
		empType2 = Radiobutton(window, text="Employer", variable=var, value=2,command=self.createSubCategory1).place(x=5,y=90)
		empType3 = Radiobutton(window, text="Fresh Graduate", variable=var, value=3, command=self.createSubCategory1).place(x=5,y=110)
		empType4 = Radiobutton(window, text="Self Employment", variable=var, value=4, command=self.createSubCategory1).place(x=5,y=130)			
		
	# SubCategory2 = Credit_Card_types
	def createSubCategory2(self):
		labelSubCat2 = Label(window, text ="Type of Credit Cards", font=('Helvetica', 10), justify=LEFT).place(x=5,y=170)
		var = IntVar()
		cardType1 = Radiobutton(window, text="Normal", variable=var, value=1,command=self.createSubCategory1).place(x=5,y=190)
		cardType2 = Radiobutton(window, text="Gold", variable=var, value=2,command=self.createSubCategory1).place(x=5,y=210)
		cardType3 = Radiobutton(window, text="Platinum", variable=var, value=3, command=self.createSubCategory1).place(x=5,y=230)			
		

	# SubCategory3 = Property_Type
	def createSubCategory3(self):
		labelSubCat3 = Label(window, text ="Type of Properties", font=('Helvetica', 10), justify=LEFT).place(x=5,y=270) 
		var = IntVar()
		propertyType1 = Radiobutton(window, text="Bungalow", variable=var, value=3, command=self.createSubCategory1).place(x=5,y=290)		
		propertyType2 = Radiobutton(window, text="Condominium", variable=var, value=1,command=self.createSubCategory1).place(x=5,y=310)
		propertyType3 = Radiobutton(window, text="Terrace", variable=var, value=2,command=self.createSubCategory1).place(x=5,y=330)
		
	# SubCategory4 = Loan_Amount
	def createSubCategory4(self):
		labelSubCat4 = Label(window, text ="Loan Amount (RM)", font=('Helvetica', 10), justify=LEFT).place(x=300,y=50)
		var = IntVar()
		loanAmount1 = Radiobutton(window, text="100,000 - 300,000", variable=var, value=1,command=self.createSubCategory1).place(x=300,y=70)
		loanAmount2 = Radiobutton(window, text="300,000 - 500,000", variable=var, value=2,command=self.createSubCategory1).place(x=300,y=90)
		loanAmount3 = Radiobutton(window, text="500,000 - 700,000", variable=var, value=3, command=self.createSubCategory1).place(x=300,y=110)
		loanAmount4 = Radiobutton(window, text="700,000 - 900,000", variable=var, value=4, command=self.createSubCategory1).place(x=300,y=130)			

	# SubCategory5 = Monthly_Salary
	def createSubCategory5(self):
		labelSubCat5 = Label(window, text ="Monthly Salary (RM)", font=('Helvetica', 10), justify=LEFT).place(x=300,y=170)
		var = IntVar()
		cardType1 = Radiobutton(window, text="4,000 - 7,000", variable=var, value=1,command=self.createSubCategory1).place(x=300,y=190)
		cardType2 = Radiobutton(window, text="7,000 - 10,000", variable=var, value=2,command=self.createSubCategory1).place(x=300,y=210)
		cardType3 = Radiobutton(window, text="10,000 - 13,000", variable=var, value=3, command=self.createSubCategory1).place(x=300,y=230)	

	def predictionButton(self):
		predictBtn = Button(window, text ="Predict Now").place(x=220,y=450) 
		
		
	# Create Machine Learning Techniques Functions
	# Decision Tree Classifier On Selected in Menu
	def runDT(self):
		dtButton = Button(window, text ="Generate Decision Tree Classifier").grid(sticky = W, column=0,row=1)
	# Naive Bayes On Selected in Menu
	def runNB(self):
		nbButton = Button(window, text ="Generate Naive Bayes").grid(sticky = W, column=0,row=1)	
	# K-Nearest Neighbour On Selected in Menu	
	def runKNN(self):
		knnButton = Button(window, text ="Generate K-Nearest Neighbour").grid(sticky = W, column=0,row=1)
	# SVM On Selected in Menu
	def runSVM(self):
		nbButton = Button(window, text ="Generate SVM").grid(sticky = W, column=0,row=1)
# Displaying the main window
window = Tk()
mainWindow = LoanPredictor(window);
window.mainloop()