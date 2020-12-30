from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter as tk

mlt1Selected = ''
mlt2Selected = ''
mlt3Selected = ''
mlt4Selected = ''
dtFrame = Frame()
knnFrame = Frame()
nbFrame = Frame()
svmFrame = Frame()

class LoanPredictor():

	def __init__(self, master): 
		# Create Main Display Window
		window.title("Bank Loan Predictor")
		window.resizable(False, False)
		window_height = 600
		window_width = 600

		screen_width = window.winfo_screenwidth()
		screen_height = window.winfo_screenheight()

		x_cordinate = int((screen_width/2) - (window_width/2))
		y_cordinate = int((screen_height/3.3) - (window_height/3.3))

		window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
		
		# Construct Menubar
		menubar = tk.Menu(window)
		window.config(menu=menubar)
		
		technicMenu = tk.Menu(menubar, tearoff = 0)
		global mlt1Selected, mlt2Selected, mlt3Selected, mlt4Selected
		mlt1Selected = BooleanVar()		
		mlt2Selected = BooleanVar()		
		mlt3Selected = BooleanVar()		
		mlt4Selected = BooleanVar()
		technicMenu.add_radiobutton(label = "Decision Tree Classifier", variable = mlt1Selected, value=True, command = self.runDT)	
		technicMenu.add_radiobutton(label = "K-Nearest Neighbour", variable = mlt2Selected, value=True, command = self.runKNN)
		technicMenu.add_radiobutton(label = "Naive Bayes", variable = mlt3Selected, value=True, command = self.runNB)
		technicMenu.add_radiobutton(label = "SVM Kernel", variable = mlt4Selected, value=True, command = self.runSVM)
		menubar.add_cascade(label = "Machine Learning Technique(s)", menu = technicMenu)	
		
		helpMenu = tk.Menu(menubar, tearoff = 0)
		helpMenu.add_command(label = "About Us", command = self.aboutUs)	
		helpMenu.add_command(label = "How to Use")			
		menubar.add_cascade(label = "Help", menu = helpMenu)
		
		self.mainDisplay()

	
	# Create Main Display
	def mainDisplay(self):
		canvas = tk.Canvas(window, bg="pink",width="600",height = "40").place(x=0,y=0)
		labelMain = tk.Label(window, bg="pink", fg="white", text ="Prediction Model", font=('Helvetica', 15, 'bold')).place(x=220,y=5)
		labelWarning = tk.Label(window, text ="Please Select a Machine Learning Technique in order to predict.", font=('Helvetica', 12)).place(x=70,y=370)
		self.createSubCategory1()
		self.createSubCategory2()
		self.createSubCategory3()
		self.createSubCategory4()
		self.createSubCategory5()
		resetBtn = Button(window, text ="Reset", command=self.resetButtonOnClicked).place(x=200,y=500, height=30, width=100) 	
		predictBtn = Button(window, text ="Predict Now", command=self.predictionButtonOnClicked).place(x=300,y=500, height=30, width=100)
		
	# Machine Learning Technique(s) Menu
	# Decision Tree Classifier On Selected in Menu
	def runDT(self):
		self.checkMLT()
		mlt1Selected.set(True)
		mlt2Selected.set(False)
		mlt3Selected.set(False)
		mlt4Selected.set(False)
		global dtFrame
		dtFrame = tk.Frame(window, width=600, height=50)
		dtFrame.place(x=210,y=400)
		dtButton = Button(dtFrame, text ="Generate Decision Tree Classifier").grid(row=1, column=0)
	# K-Nearest Neighbour On Selected in Menu	
	def runKNN(self):
		self.checkMLT()
		mlt1Selected.set(False)
		mlt2Selected.set(True)
		mlt3Selected.set(False)
		mlt4Selected.set(False)
		global knnFrame
		knnFrame = tk.Frame(window, width=600, height=50)
		knnFrame.place(x=210,y=400)
		knnSelector = Scale(knnFrame, from_=0, to=9, orient=HORIZONTAL).grid(row=0, column=0)
		knnButton = Button(knnFrame, text ="Generate K-Nearest Neighbour").grid(row=1, column=0)
	# Naive Bayes On Selected in Menu
	def runNB(self):
		self.checkMLT()
		mlt1Selected.set(False)
		mlt2Selected.set(False)
		mlt3Selected.set(True)
		mlt4Selected.set(False)	
		global nbFrame
		nbFrame = tk.Frame(window, width=600, height=50)
		nbFrame.place(x=210,y=400)
		nbButton = Button(nbFrame, text ="Generate Naive Bayes").grid(row=1, column=0)
	# SVM On Selected in Menu
	def runSVM(self):
		self.checkMLT()
		mlt1Selected.set(False)
		mlt2Selected.set(False)
		mlt3Selected.set(False)
		mlt4Selected.set(True)	
		global svmFrame
		svmFrame = tk.Frame(window, width=600, height=50)
		svmFrame.place(x=210,y=400)
		var = StringVar()
		kernel1 = Radiobutton(svmFrame, text="Rbf", variable=var, value=1,command=self.SVMRbf).grid(row=0, column=0)
		kernel2 = Radiobutton(svmFrame, text="Linear", variable=var, value=2,command=self.SVMLinear).grid(row=1, column=0)
		kernel3 = Radiobutton(svmFrame, text="Polynomial", variable=var, value=3, command=self.SVMPoly).grid(row=2, column=0)
		svmButton = Button(svmFrame, text ="Generate SVM").grid(row=3, column=0)
	# Check if the MLT is Clicked/Selected
	def checkMLT(self):
		if mlt1Selected == True:
			for widget in knnFrame.winfo_children():
			   widget.destroy()
			knnFrame.pack_forget()   
			for widget in nbFrame.winfo_children():
			   widget.destroy()
			nbFrame.pack_forget() 
			for widget in svmFrame.winfo_children():
			   widget.destroy()
			svmFrame.pack_forget() 
		if mlt2Selected == True:
			for widget in dtFrame.winfo_children():
			   widget.destroy()
			dtFrame.pack_forget()   
			for widget in nbFrame.winfo_children():
			   widget.destroy()
			nbFrame.pack_forget() 
			for widget in svmFrame.winfo_children():
			   widget.destroy()
			svmFrame.pack_forget() 
		if mlt3Selected == True:
			for widget in dtFrame.winfo_children():
			   widget.destroy()
			dtFrame.pack_forget()   
			for widget in knnFrame.winfo_children():
			   widget.destroy()
			knnFrame.pack_forget() 
			for widget in svmFrame.winfo_children():
			   widget.destroy()
			svmFrame.pack_forget() 
		if mlt4Selected == True:
			for widget in dtFrame.winfo_children():
			   widget.destroy()
			dtFrame.pack_forget()   
			for widget in knnFrame.winfo_children():
			   widget.destroy()
			knnFrame.pack_forget() 
			for widget in nbFrame.winfo_children():
			   widget.destroy()
			nbFrame.pack_forget() 	
		else:
			for widget in dtFrame.winfo_children():
			   widget.destroy()
			dtFrame.pack_forget()   
			for widget in knnFrame.winfo_children():
			   widget.destroy()
			knnFrame.pack_forget() 
			for widget in nbFrame.winfo_children():
			   widget.destroy()
			nbFrame.pack_forget() 		
			for widget in svmFrame.winfo_children():
			   widget.destroy()
			svmFrame.pack_forget() 			
	# Help Menu
	def aboutUs(self):
		messagebox.showinfo("About Us","TDS3301 - Data Mining Project\nGroup Member:\nOng Shuoh Chwen 1171102212\nYong Wen Kai 1171101664\nLecturer:\nDr. Ting Choo Yee")
	
	def howToUse(self):
		pass
	# Create Sub-Categories
	# SubCategory1 = Employment_Type
	def createSubCategory1(self):
		labelSubCat1 = Label(window, text ="Employment Type", font=('Helvetica', 10), justify=LEFT).place(x=10,y=50)
		var = StringVar()
		empType1 = Radiobutton(window, text="Employee", variable=var, value=1,command=self.saveSelectedValues).place(x=10,y=70)
		empType2 = Radiobutton(window, text="Employer", variable=var, value=2,command=self.saveSelectedValues).place(x=10,y=90)
		empType3 = Radiobutton(window, text="Fresh Graduate", variable=var, value=3, command=self.saveSelectedValues).place(x=10,y=110)
		empType4 = Radiobutton(window, text="Self Employment", variable=var, value=4, command=self.saveSelectedValues).place(x=10,y=130)			
		
	# SubCategory2 = Credit_Card_types
	def createSubCategory2(self):
		labelSubCat2 = Label(window, text ="Type of Credit Cards", font=('Helvetica', 10), justify=LEFT).place(x=10,y=170)
		var = StringVar()
		cardType1 = Radiobutton(window, text="Normal", variable=var, value=1,command=self.saveSelectedValues).place(x=10,y=190)
		cardType2 = Radiobutton(window, text="Gold", variable=var, value=2,command=self.saveSelectedValues).place(x=10,y=210)
		cardType3 = Radiobutton(window, text="Platinum", variable=var, value=3, command=self.saveSelectedValues).place(x=10,y=230)			
		
	# SubCategory3 = Property_Type
	def createSubCategory3(self):
		labelSubCat3 = Label(window, text ="Type of Properties", font=('Helvetica', 10), justify=LEFT).place(x=10,y=270) 
		var = StringVar()
		propertyType1 = Radiobutton(window, text="Bungalow", variable=var, value=3, command=self.saveSelectedValues).place(x=10,y=290)		
		propertyType2 = Radiobutton(window, text="Condominium", variable=var, value=1,command=self.saveSelectedValues).place(x=10,y=310)
		propertyType3 = Radiobutton(window, text="Terrace", variable=var, value=2,command=self.saveSelectedValues).place(x=10,y=330)
		
	# SubCategory4 = Loan_Amount
	def createSubCategory4(self):
		labelSubCat4 = Label(window, text ="Loan Amount (RM)", font=('Helvetica', 10), justify=LEFT).place(x=350,y=50)
		var = StringVar()
		loanAmount1 = Radiobutton(window, text="100,000 - 300,000", variable=var, value=1,command=self.saveSelectedValues).place(x=350,y=70)
		loanAmount2 = Radiobutton(window, text="300,000 - 500,000", variable=var, value=2,command=self.saveSelectedValues).place(x=350,y=90)
		loanAmount3 = Radiobutton(window, text="500,000 - 700,000", variable=var, value=3, command=self.saveSelectedValues).place(x=350,y=110)
		loanAmount4 = Radiobutton(window, text="700,000 - 900,000", variable=var, value=4, command=self.saveSelectedValues).place(x=350,y=130)			

	# SubCategory5 = Monthly_Salary
	def createSubCategory5(self):
		labelSubCat5 = Label(window, text ="Monthly Salary (RM)", font=('Helvetica', 10), justify=LEFT).place(x=350,y=170)
		var = StringVar()
		cardType1 = Radiobutton(window, text="4,000 - 7,000", variable=var, value=1,command=self.saveSelectedValues).place(x=350,y=190)
		cardType2 = Radiobutton(window, text="7,000 - 10,000", variable=var, value=2,command=self.saveSelectedValues).place(x=350,y=210)
		cardType3 = Radiobutton(window, text="10,000 - 13,000", variable=var, value=3, command=self.saveSelectedValues).place(x=350,y=230)	

	# Reset Button On Clicked
	def resetButtonOnClicked(self):
		self.mainDisplay()
		
	# Prediction Button On Clicked
	def predictionButtonOnClicked(self):
		pass
	
	def saveSelectedValues(self):
		pass
		#employmentType = 
	# Generate SVM Accordingly
	def SVMRbf(self): 
		pass
	def SVMLinear(self): 
		pass
	def SVMPoly(self): 
		pass
# Displaying the main window
window = Tk()
mainWindow = LoanPredictor(window);
window.mainloop()