from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter as tk
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from apyori import apriori
import altair as alt
import missingno as msno
from string import ascii_letters
import seaborn as sns
import imblearn
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
#from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.neighbors import KNeighborsClassifier
from kmodes.kmodes import KModes
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
		
class LoanPredictor():

	def __init__(self): 
		# Create Main Display Window
		window = tk.Tk()
		self.dtFrame = Frame()
		self.knnFrame = Frame()
		self.nbFrame = Frame()
		self.svmFrame = Frame()
		self.predictionFrame = Frame()
		self.b4SmoteFrame = Frame()
		self.smoteFrame = Frame()
		self.armFrame = Frame()
		self.genArmFrame = Frame()
		self.kmcFrame = Frame()
		self.employmentType = ''
		self.propertyType =''
		self.cardType = ''
		self.mthSalary = ''
		self.loanAmount = ''
		self.window = window
		self.window.title("Bank Loan Predictor")
		self.window.resizable(False, False)
		window_height = 600
		window_width = 1200

		screen_width = self.window.winfo_screenwidth()
		screen_height = self.window.winfo_screenheight()

		x_cordinate = int((screen_width/3) - (window_width/3))
		y_cordinate = int((screen_height/3) - (window_height/3))

		self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
		
		# Construct Menubar
		menubar = tk.Menu(self.window)
		self.window.config(menu=menubar)
		
		self.armSelected = BooleanVar()
		self.genArmSelected = BooleanVar()
		menubar.add_radiobutton(label = "Association Rule Mining", variable = self.armSelected, value=True, command=self.runARM)
		
		visualMenu = tk.Menu(menubar, tearoff = 0)
		self.b4SmoteSelected = BooleanVar()
		self.smoteSelected = BooleanVar()
		visualMenu.add_radiobutton(label="Before SMOTE", variable = self.b4SmoteSelected, value=True, command=self.runDf2)
		visualMenu.add_separator()
		visualMenu.add_radiobutton(label="After SMOTE", variable = self.smoteSelected, value=True, command=self.runDfSmote)
		menubar.add_cascade(label = "EDA", menu = visualMenu)
		
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
		
		clusterMenu = tk.Menu(menubar, tearoff = 0)
		self.kmcSelected = BooleanVar()
		clusterMenu.add_radiobutton(label = "K Mode Clustering", variable = self.kmcSelected, value=True, command=self.runkmc)
		menubar.add_cascade(label = "Clustering", menu = clusterMenu)
		
		self.pmSelected = BooleanVar()		
		menubar.add_radiobutton(label = "Prediction Model", variable = self.pmSelected, value=True, command=self.runPM)
		
		helpMenu = tk.Menu(menubar, tearoff = 0)
		helpMenu.add_command(label = "About Us", command = self.aboutUs)	
		helpMenu.add_command(label = "How to Use", command = self.howToUse)			
		menubar.add_cascade(label = "Help", menu = helpMenu)
		
		self.runPM()
		self.dataPreprocess()
	
	# Create Main Display
	def runPM(self):
		self.destroyFrames()
		self.predictionFrame = tk.Frame(self.window, width=1200, height=600)
		self.predictionFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.predictionFrame, bg="pink",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.predictionFrame, bg="pink", fg="white", text ="Prediction Model", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.predictionFrame, width="1200",height = "600").grid(row=1, column=0)	
		# Create Sub-Categories
		# SubCategory1 = Employment_Type
		labelSubCat1 = Label(self.predictionFrame, text ="Employment Type", font=('Helvetica', 10), justify=LEFT).place(x=10,y=50)
		self.employmentType = StringVar()
		empType1 = Radiobutton(self.predictionFrame, text="Employee", variable=self.employmentType, value="Employee",command=self.saveSelectedValues).place(x=10,y=70)
		empType2 = Radiobutton(self.predictionFrame, text="Employer", variable=self.employmentType, value="Employer",command=self.saveSelectedValues).place(x=10,y=90)
		empType3 = Radiobutton(self.predictionFrame, text="Fresh Graduate", variable=self.employmentType, value="Fresh Graduate", command=self.saveSelectedValues).place(x=10,y=110)
		empType4 = Radiobutton(self.predictionFrame, text="Self Employment", variable=self.employmentType, value="Self Employment", command=self.saveSelectedValues).place(x=10,y=130)			
		
		# SubCategory2 = Credit_Card_types
		labelSubCat2 = Label(self.predictionFrame, text ="Type of Credit Cards", font=('Helvetica', 10), justify=LEFT).place(x=10,y=170)
		self.cardType = StringVar()
		cardType1 = Radiobutton(self.predictionFrame, text="Normal", variable=self.cardType, value="Normal",command=self.saveSelectedValues).place(x=10,y=190)
		cardType2 = Radiobutton(self.predictionFrame, text="Gold", variable=self.cardType, value="Gold",command=self.saveSelectedValues).place(x=10,y=210)
		cardType3 = Radiobutton(self.predictionFrame, text="Platinum", variable=self.cardType, value="Platinum", command=self.saveSelectedValues).place(x=10,y=230)	
		
		# SubCategory3 = Property_Type
		labelSubCat3 = Label(self.predictionFrame, text ="Type of Properties", font=('Helvetica', 10), justify=LEFT).place(x=10,y=270) 
		self.propertyType = StringVar()
		propertyType1 = Radiobutton(self.predictionFrame, text="Bungalow", variable=self.propertyType, value="Bungalow", command=self.saveSelectedValues).place(x=10,y=290)		
		propertyType2 = Radiobutton(self.predictionFrame, text="Condominium", variable=self.propertyType, value="Condominium",command=self.saveSelectedValues).place(x=10,y=310)
		propertyType3 = Radiobutton(self.predictionFrame, text="Flat", variable=self.propertyType, value="Flat",command=self.saveSelectedValues).place(x=10,y=330)
		propertyType4 = Radiobutton(self.predictionFrame, text="Terrace", variable=self.propertyType, value="Terrace",command=self.saveSelectedValues).place(x=10,y=350)
		
		# SubCategory4 = Loan_Amount
		labelSubCat4 = Label(self.predictionFrame, text ="Loan Amount (RM)", font=('Helvetica', 10), justify=LEFT).place(x=350,y=50)
		self.loanAmount = StringVar()
		loanAmount1 = Radiobutton(self.predictionFrame, text="100,000 - 300,000", variable=self.loanAmount, value="100,000 - 300,000", command=self.saveSelectedValues).place(x=350,y=70)
		loanAmount2 = Radiobutton(self.predictionFrame, text="300,000 - 500,000", variable=self.loanAmount, value="300,000 - 500,000", command=self.saveSelectedValues).place(x=350,y=90)
		loanAmount3 = Radiobutton(self.predictionFrame, text="500,000 - 700,000", variable=self.loanAmount, value="500,000 - 700,000", command=self.saveSelectedValues).place(x=350,y=110)
		loanAmount4 = Radiobutton(self.predictionFrame, text="700,000 - 900,000", variable=self.loanAmount, value="700,000 - 900,000", command=self.saveSelectedValues).place(x=350,y=130)
		
		# SubCategory5 = Monthly_Salary
		labelSubCat5 = Label(self.predictionFrame, text ="Monthly Salary (RM)", font=('Helvetica', 10), justify=LEFT).place(x=350,y=170)
		self.mthSalary = StringVar()
		mthSalary1 = Radiobutton(self.predictionFrame, text="<4,000", variable=self.mthSalary, value="<4,000", command=self.saveSelectedValues).place(x=350,y=190)
		mthSalary2 = Radiobutton(self.predictionFrame, text="4,000 - 7,000", variable=self.mthSalary, value="4,000 - 7,000", command=self.saveSelectedValues).place(x=350,y=210)
		mthSalary3 = Radiobutton(self.predictionFrame, text="7,000 - 10,000", variable=self.mthSalary, value="7,000 - 10,000", command=self.saveSelectedValues).place(x=350,y=230)
		mthSalary4 = Radiobutton(self.predictionFrame, text="10,000 - 13,000", variable=self.mthSalary, value="10,000 - 13,000", command=self.saveSelectedValues).place(x=350,y=250)	
		
		resetBtn = Button(self.predictionFrame, text ="Reset", command=self.resetButtonOnClicked).place(x=200,y=500, height=30, width=100) 	
		predictBtn = Button(self.predictionFrame, text ="Predict Now", command=self.predictionButtonOnClicked).place(x=300,y=500, height=30, width=100)
		
	def dataPreprocess(self):
		# Load Data
		df = pd.read_csv("Bank_CS.csv")
		df1 = df.copy()
		df1
		
		# Duplicate the dataframe
		df1 = df.copy()
		
		# Drop the unknown column ("...") 
		df1.drop(df1.iloc[:,10:11], inplace=True, axis=1)
		
		# Check missing values
		df1.isnull().sum()
		msno.bar(df1)
		
		# Fill all missing values with mode
		for column in df1.columns:
			df1[column].fillna(df1[column].mode()[0], inplace=True)
			
		# Check missing values again
		df1.isnull().sum()
		msno.bar(df1)
		
		# Dealing with Noisy Data
		# Employment_Type
		df1.Employment_Type = df1.Employment_Type.replace("employer", "Employer", regex=True)
		df1.Employment_Type = df1.Employment_Type.replace("Self_Employed", "Self Employed", regex=True)
		df1.Employment_Type = df1.Employment_Type.replace("government", "Government", regex=True)
		df1.Employment_Type = df1.Employment_Type.replace("employee", "Employee", regex=True)
		df1.Employment_Type = df1.Employment_Type.replace("Fresh_Graduate", "Fresh Graduate", regex=True)

		# Credit_Card_types
		df1.Credit_Card_types = df1.Credit_Card_types.replace("platinum", "Platinum", regex=True)
		df1.Credit_Card_types = df1.Credit_Card_types.replace("normal", "Normal", regex=True)
		df1.Credit_Card_types = df1.Credit_Card_types.replace("gold", "Gold", regex=True)

		# Property_Type
		df1.Property_Type = df1.Property_Type.replace("bungalow", "Bungalow", regex=True)
		df1.Property_Type = df1.Property_Type.replace("condominium", "Condominium", regex=True)
		df1.Property_Type = df1.Property_Type.replace("flat", "Flat", regex=True)
		df1.Property_Type = df1.Property_Type.replace("terrace", "Terrace", regex=True)

		# State
		df1.State = df1.State.replace("P.Pinang", "Penang", regex=True)
		df1.State = df1.State.replace("Pulau Penang", "Penang", regex=True)
		df1.State = df1.State.replace("Johor B", "Johor", regex=True)
		df1.State = df1.State.replace("K.L", "Kuala Lumpur", regex=True)
		df1.State = df1.State.replace("N.Sembilan", "Negeri Sembilan", regex=True)
		df1.State = df1.State.replace("N.S", "Negeri Sembilan", regex=True)
		df1.State = df1.State.replace("SWK", "Sarawak", regex=True)
		df1.State = df1.State.replace("Trengganu", "Terrenganu", regex=True)
		
		#Duplicate the dataframe again for replacing dtype
		self.df2 = df1.copy()
		
		# Change Loan_Amount to Categorical DType
		self.df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(100000.0,300000.0), '100000-300000', self.df2['Loan_Amount'])
		self.df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(300000.0,500000.0), '300000-500000', self.df2['Loan_Amount'])
		self.df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(500000.0,700000.0), '500000-700000', self.df2['Loan_Amount'])
		self.df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(700000.0,900000.0), '700000-900000', self.df2['Loan_Amount'])
		self.df2['Loan_Amount'] = self.df2['Loan_Amount'].astype("category")

		# Change Loan_Amount to Categorical DType
		self.df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'] < 4000.0, '<4000', self.df2['Monthly_Salary'])
		self.df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'].between(4000.0,7000.0), '4000-7000', self.df2['Monthly_Salary'])
		self.df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'].between(7000.0,10000.0), '7000-10000', self.df2['Monthly_Salary'])
		self.df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'].between(10000.0,13000.0), '10000-13000', self.df2['Monthly_Salary'])
		self.df2['Monthly_Salary'] = self.df2['Monthly_Salary'].astype("category")
		
		#Applying SMOTE based on Decision
		self.df3 = self.df2.copy()

		self.dictionary = defaultdict(LabelEncoder)
		self.df3 = self.df2.apply(lambda x: self.dictionary[x.name].fit_transform(x))

		y = self.df3.Decision
		X = self.df3.drop(columns =['Decision'])
		
		smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
		self.X_res, self.y_res = smt.fit_resample(X, y)
		colnames = self.X_res.columns
		
		# Exploratory Data Analysis after SMOTE (Based on Decision)
		self.dfSmote = pd.concat([self.X_res.reset_index(drop=True), self.y_res], axis=1) 
		self.dfSmote = self.dfSmote.apply(lambda x: self.dictionary[x.name].inverse_transform(x))
		
		# Remove the features with lowest ranking after performing Feature Selection (Boruta and RFE)
		self.X_res.drop(columns=["Number_of_Properties","Loan_Amount"], axis=1, inplace=True)
		
		# Machine Learning Techiniques
		# Support Vector Machine (SVM)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_res, self.y_res, test_size=0.3,random_state=1)
		
		self.dictionary1 = defaultdict(LabelEncoder)
		self.dfSmote1 = self.dfSmote.apply(lambda x: self.dictionary1[x.name].fit_transform(x))
	
	# Visualizing Comparison Charts from Exploratory Data Analysis (Before SMOTE and after SMOTE)
	def runDf2(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(True)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.b4SmoteFrame = tk.Frame(self.window, width=1200, height=600)
		self.b4SmoteFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.b4SmoteFrame, bg="pink",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.b4SmoteFrame, bg="pink", fg="white", text ="EDA (Before SMOTE)", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.b4SmoteFrame, width="1200",height = "600").grid(row=1, column=0)
		
		# which type of employment is likely to have the loan accepted?
		typeOfEmploy = DataFrame(self.df3, columns=['Employment_Type','Decision'])
		
		figure = plt.Figure(figsize=(5,5), dpi=50)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.b4SmoteFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=10,y=60)
		typeOfEmploy = typeOfEmploy[['Employment_Type', 'Decision']].groupby('Employment_Type').sum()
		typeOfEmploy.plot(kind='bar', legend=True, ax=ax)
		ax.set_title('Type of employment to have the loan accepted')
		
		# which type of credit card user is likely to have the loan accepted?
		typeOfCard = DataFrame(self.df3, columns=['Credit_Card_types','Decision'])
		
		figure1 = plt.Figure(figsize=(5,5), dpi=50)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.b4SmoteFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=270,y=60)
		typeOfCard = typeOfCard[['Credit_Card_types', 'Decision']].groupby('Credit_Card_types').sum()
		typeOfCard.plot(kind='bar', legend=True, ax=ax1)
		ax1.set_title('Type of credit card user to have the loan accepted')	

		# which type of properties is likely to have the loan accepted?
		typeOfProperty = DataFrame(self.df3, columns=['Property_Type','Decision'])
		
		figure2 = plt.Figure(figsize=(5,5), dpi=50)
		ax2 = figure2.add_subplot(111)
		canvas2 = FigureCanvasTkAgg(figure2,master=self.b4SmoteFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=530,y=60)
		typeOfProperty = typeOfProperty[['Property_Type', 'Decision']].groupby('Property_Type').sum()
		typeOfProperty.plot(kind='bar', legend=True, ax=ax2)
		ax2.set_title('Type of properties to have the loan accepted')		
		
		# what is the monthly salary that is likely to have the loan accepted?
		monthlySalary = DataFrame(self.df3, columns=['Monthly_Salary','Decision'])
		
		figure3 = plt.Figure(figsize=(5,5), dpi=50)
		ax3 = figure3.add_subplot(111)
		canvas3 = FigureCanvasTkAgg(figure3,master=self.b4SmoteFrame)
		canvas3.draw()
		canvas3.get_tk_widget().place(x=10,y=320)
		monthlySalary  = monthlySalary[['Monthly_Salary', 'Decision']].groupby('Monthly_Salary').sum()
		monthlySalary.plot(kind='bar', legend=True, ax=ax3)
		ax3.set_title('Monthly salary to have the loan accepted')	

		# Count the number of customers by Decision and Employment_Type
		employVsDecision = DataFrame(self.df3, columns=['Employment_Type','Decision'])
		
		figure4 = plt.Figure(figsize=(5,5), dpi=50)
		ax4 = figure4.add_subplot(111)
		canvas4 = FigureCanvasTkAgg(figure4,master=self.b4SmoteFrame)
		canvas4.draw()
		canvas4.get_tk_widget().place(x=270,y=320)
		employVsDecision  = employVsDecision[['Employment_Type', 'Decision']].groupby('Decision').sum()
		employVsDecision.plot(kind='bar', legend=True, ax=ax4)
		ax4.set_title('Number of customers by Decision and Employment_Type')	
		
		# what is the monthly salary that is likely to have the loan accepted?
		salaryVsDecision = DataFrame(self.df3, columns=['Monthly_Salary','Decision'])
		
		figure5 = plt.Figure(figsize=(5,5), dpi=50)
		ax5 = figure5.add_subplot(111)
		canvas5 = FigureCanvasTkAgg(figure5,master=self.b4SmoteFrame)
		canvas5.draw()
		canvas5.get_tk_widget().place(x=530,y=320)
		salaryVsDecision  = salaryVsDecision[['Monthly_Salary', 'Decision']].groupby('Decision').sum()
		salaryVsDecision.plot(kind='bar', legend=True, ax=ax5)
		ax5.set_title('Number of customers by Decision and Monthly_Salary')	
		
		# what is the decision made by the bank the most frequent?
		decision = DataFrame(self.df2, columns=['Decision'])
	
		figure6 = plt.Figure(figsize=(8,10), dpi=50)
		ax6 = figure6.add_subplot(111)
		canvas6 = FigureCanvasTkAgg(figure6,master=self.b4SmoteFrame)
		canvas6.draw()
		canvas6.get_tk_widget().place(x=790,y=65)

		decision['Decision'].value_counts().plot(kind='bar', legend=True, ax=ax6, color=tuple(["g", "r","b","y","k"]))
		ax6.set_title('Most frequent decision made by the bank')
		ax6.set_xlabel('Decision')		

		
	def runDfSmote(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(True)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.SmoteFrame = tk.Frame(self.window, width=1200, height=600)
		self.SmoteFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.SmoteFrame, bg="pink",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.SmoteFrame, bg="pink", fg="white", text ="EDA (After SMOTE)", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.SmoteFrame, width="1200",height = "600").grid(row=1, column=0)
		
		# which type of employment is likely to have the loan accepted?
		typeOfEmploy = DataFrame(self.dfSmote1, columns=['Employment_Type','Decision'])
		
		figure = plt.Figure(figsize=(5,5), dpi=50)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.SmoteFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=10,y=60)
		typeOfEmploy['Employment_Type'].value_counts().plot(kind='bar', legend=True, ax=ax)
		ax.set_title('Type of employment to have the loan accepted')
		ax.legend(["Employer", "Self-Employed", "Government", "Employee", "Fresh Graduate"])
	
		# which type of credit card user is likely to have the loan accepted?
		typeOfCard = DataFrame(self.dfSmote1, columns=['Credit_Card_types','Decision'])
		
		figure1 = plt.Figure(figsize=(5,5), dpi=50)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.SmoteFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=270,y=60)
		typeOfCard = typeOfCard[['Credit_Card_types', 'Decision']].groupby('Credit_Card_types').sum()
		typeOfCard.plot(kind='bar', legend=True, ax=ax1)
		ax1.set_title('Type of credit card user to have the loan accepted')	

		# which type of properties is likely to have the loan accepted?
		typeOfProperty = DataFrame(self.dfSmote1, columns=['Property_Type','Decision'])
		
		figure2 = plt.Figure(figsize=(5,5), dpi=50)
		ax2 = figure2.add_subplot(111)
		canvas2 = FigureCanvasTkAgg(figure2,master=self.SmoteFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=530,y=60)
		typeOfProperty = typeOfProperty[['Property_Type', 'Decision']].groupby('Property_Type').sum()
		typeOfProperty.plot(kind='bar', legend=True, ax=ax2)
		ax2.set_title('Type of properties to have the loan accepted')		
		
		# what is the monthly salary that is likely to have the loan accepted?
		monthlySalary = DataFrame(self.dfSmote1, columns=['Monthly_Salary','Decision'])
		
		figure3 = plt.Figure(figsize=(5,5), dpi=50)
		ax3 = figure3.add_subplot(111)
		canvas3 = FigureCanvasTkAgg(figure3,master=self.SmoteFrame)
		canvas3.draw()
		canvas3.get_tk_widget().place(x=10,y=320)
		monthlySalary  = monthlySalary[['Monthly_Salary', 'Decision']].groupby('Monthly_Salary').sum()
		monthlySalary.plot(kind='bar', legend=True, ax=ax3)
		ax3.set_title('Monthly salary to have the loan accepted')	

		# Count the number of customers by Decision and Employment_Type
		employVsDecision = DataFrame(self.dfSmote1, columns=['Employment_Type','Decision'])
		
		figure4 = plt.Figure(figsize=(5,5), dpi=50)
		ax4 = figure4.add_subplot(111)
		canvas4 = FigureCanvasTkAgg(figure4,master=self.SmoteFrame)
		canvas4.draw()
		canvas4.get_tk_widget().place(x=270,y=320)
		employVsDecision  = employVsDecision[['Employment_Type', 'Decision']].groupby('Decision').sum()
		employVsDecision.plot(kind='bar', legend=True, ax=ax4)
		ax4.set_title('Number of customers by Decision and Employment_Type')	
		
		# what is the monthly salary that is likely to have the loan accepted?
		salaryVsDecision = DataFrame(self.dfSmote1, columns=['Monthly_Salary','Decision'])
		
		figure5 = plt.Figure(figsize=(5,5), dpi=50)
		ax5 = figure5.add_subplot(111)
		canvas5 = FigureCanvasTkAgg(figure5,master=self.SmoteFrame)
		canvas5.draw()
		canvas5.get_tk_widget().place(x=530,y=320)
		salaryVsDecision  = salaryVsDecision[['Monthly_Salary', 'Decision']].groupby('Decision').sum()
		salaryVsDecision.plot(kind='bar', legend=True, ax=ax5)
		ax5.set_title('Number of customers by Decision and Monthly_Salary')	
		
		# what is the decision made by the bank the most frequent?
		decision = DataFrame(self.dfSmote1, columns=['Decision'])
	
		figure6 = plt.Figure(figsize=(8,10), dpi=50)
		ax6 = figure6.add_subplot(111)
		canvas6 = FigureCanvasTkAgg(figure6,master=self.SmoteFrame)
		canvas6.draw()
		canvas6.get_tk_widget().place(x=790,y=65)

		decision['Decision'].value_counts().plot(kind='bar', legend=True, ax=ax6, color=tuple(["g", "r","b","y","k"]))
		ax6.set_title('Most frequent decision made by the bank')
		ax6.set_xlabel('Decision')
		ax6.legend(["Reject", "Accept"])
		#ax6.plot(x='Decision', rot=0, title='', figsize=(15,10), fontsize=12)
		#ax6.set_index('Decision').plot.bar(rot=0, title='Most frequent decision made by the bank', figsize=(15,10), fontsize=12)
		
	# Association Rule Mining
	def runARM(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(True)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.armFrame = tk.Frame(self.window, width=1200, height=600)
		self.armFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.armFrame, bg="pink",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.armFrame, bg="pink", fg="white", text ="Association Rule Mining", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.armFrame, width="1200",height = "600").grid(row=1, column=0)

		self.df4 = self.df3.copy()
		self.df4.drop(axis= 1, inplace = True, columns = ['Decision','Credit_Card_Exceed_Months','Loan_Amount','Loan_Tenure_Year','More_Than_One_Products','Number_of_Dependents','Years_to_Financial_Freedom','Number_of_Credit_Card_Facility','Number_of_Properties','Number_of_Loan_to_Approve','Years_for_Property_to_Completion','State','Number_of_Side_Income','Total_Sum_of_Loan','Total_Income_for_Join_Application','Score'])
		self.df4 = self.df4.apply(lambda x: self.dictionary[x.name].inverse_transform(x))
		
		self.minSuppValue = DoubleVar()
		self.minConfValue = DoubleVar()
		self.minLiftValue = DoubleVar()
		self.maxAnteValue = DoubleVar()
		self.maxConsValue = DoubleVar()
		
		minSuppLabel = Label(self.armFrame, text='Choose the minimum support').place(x=10,y=50)
		minsupp = tk.Scale(self.armFrame, from_=0.0001, to=0.0300, digits = 5, resolution = 0.0001, orient=HORIZONTAL, variable=self.minSuppValue).place(x=10,y=70)
		minConfLabel = Label(self.armFrame, text='Choose the minimum confidence').place(x=10,y=110)		
		minconf = tk.Scale(self.armFrame, from_=0.50, to=1.00, digits = 3, resolution = 0.01, orient=HORIZONTAL, variable=self.minConfValue).place(x=10,y=130)
		minLiftLabel = Label(self.armFrame, text='Choose the minimum lift').place(x=10,y=170)
		minlift = tk.Scale(self.armFrame, from_=0.50, to=2.00, digits = 3, resolution = 0.01, orient=HORIZONTAL, variable=self.minLiftValue).place(x=10,y=190)
		maxAnteLabel = Label(self.armFrame, text='Choose maximum number of antecedent').place(x=10,y=230)		
		maxante = tk.Scale(self.armFrame, from_=1, to=10, orient=HORIZONTAL, variable=self.maxAnteValue).place(x=10,y=250)
		maxConsLabel = Label(self.armFrame, text='Choose maximum number of consequent').place(x=10,y=290)			
		maxcons = tk.Scale(self.armFrame, from_=1, to=10, orient=HORIZONTAL, variable=self.maxConsValue).place(x=10,y=310)
		generateBtn = Button(self.armFrame, text="Generate Rules", command=self.generateARM).place(x=550,y=500, height=50, width=150)
		
	def generateARM(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(True)
		self.kmcSelected.set(False)
		self.genArmFrame = tk.Frame(self.window, width=1200, height=600)
		self.genArmFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.genArmFrame, bg="pink",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.genArmFrame, bg="pink", fg="white", text ="Association Rule Mining", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.genArmFrame, width="1200",height = "600").grid(row=1, column=0)
		resetBtn = Button(self.genArmFrame, text="Reset", command=self.runARM).place(x=230,y=350, height=50, width=150)
		
		
		basketEmployer = (self.df3[self.df3['Employment_Type'] =="Employer"] 
          .groupby(['Employment_Type', 'Monthly_Salary'])['Score'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('Employment_Type')
		  ) 
		basketSelfEmp = (self.df3[self.df3['Employment_Type'] =="Self-Employed"] 
          .groupby(['Employment_Type', 'Monthly_Salary'])['Score'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('Employment_Type')
		  ) 
		basketGov = (self.df3[self.df3['Employment_Type'] =="Government"] 
          .groupby(['Employment_Type', 'Monthly_Salary'])['Score'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('Employment_Type')
		  ) 
		basketEmployee = (self.df3[self.df3['Employment_Type'] =="Employee"] 
          .groupby(['Employment_Type', 'Monthly_Salary'])['Score'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('Employment_Type')
		  ) 		  
		basketFreshGrad = (self.df3[self.df3['Employment_Type'] =="Fresh Graduate"] 
          .groupby(['Employment_Type', 'Monthly_Salary'])['Score'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('Employment_Type')
		  ) 
		 
		# Defining the hot encoding function to make the data suitable for the concerned libraries
		
		# Encoding the datasets 
		basket_encoded = basketEmployer.applymap(self.hot_encode) 
		basketEmployer = basket_encoded 
		print (basketEmployer)
		print ("yeet")
		  
		basket_encoded = basketSelfEmp.applymap(self.hot_encode) 
		basketSelfEmp = basket_encoded 
		  
		basket_encoded = basketGov.applymap(self.hot_encode) 
		basketGov = basket_encoded 
		  
		basket_encoded = basketEmployee.applymap(self.hot_encode) 
		basketEmployee = basket_encoded 		
		
		basket_encoded = basketFreshGrad.applymap(self.hot_encode) 
		basketFreshGrad = basket_encoded
		
		records = []
		for i in range(0, 2350):
			records.append([str(self.df4.values[i,j]) for j in range(0, 4)])
		
		print (records)
		association_rules = apriori(records, min_support=self.minSuppValue.get(), min_confidence=self.minConfValue.get(), min_lift=self.minLiftValue.get(), min_length=2)
		
		# Building the model 
		frq_items = apriori(basketEmployer, min_support = 0.05, use_colnames = True) 
		print (frq_items)
		print ("123")
		  
		# Collecting the inferred rules in a dataframe 
		#rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
		#print(rules.head()) 
		
		association_rules_list = list(association_rules)
		labelRules = tk.Label(self.genArmFrame, text = "Total Rules Generated: " + str(len(association_rules_list)), font=('Helvetica', 15, 'bold')).place(x=100,y=100)
		labelMinSupp = tk.Label(self.genArmFrame, text = "Minimum Support Value: " + str(self.minSuppValue.get()), font=('Helvetica', 12)).place(x=100,y=130)
		labelMinConf = tk.Label(self.genArmFrame, text = "Minimum Confidence Value: " + str(self.minConfValue.get()), font=('Helvetica', 12)).place(x=100,y=150)	
		labelMinLift = tk.Label(self.genArmFrame, text = "Minimum Lift Value: " + str(self.minLiftValue.get()), font=('Helvetica', 12)).place(x=100,y=170)	
		labelMaxAnte = tk.Label(self.genArmFrame, text = "Maximum Antecedent Value: " + str(self.maxAnteValue.get()), font=('Helvetica', 12)).place(x=100,y=190)	
		labelMaxCons = tk.Label(self.genArmFrame, text = "Maximum Consequent Value: " + str(self.maxConsValue.get()), font=('Helvetica', 12)).place(x=100,y=210)	
		labelTop10Rules = tk.Label(self.genArmFrame, text = "Top 10 Rules", font=('Helvetica', 15, 'bold')).place(x=500,y=50)	
		'''
		support=DataFrame(rules, columns=['support'])
		confidence=DataFrame(rules, columns=['confidence'])
		
		for i in range (len(support)):
		   support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
		   confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
		 
		plt.scatter(support, confidence,   alpha=0.5, marker="*")
		plt.xlabel('support')
		plt.ylabel('confidence') 
		plt.show()
		'''
		#print(len(association_rules_list))
		
		for item in association_rules_list:

			# first index of the inner list
			# Contains base item and add item
			pair = item[0] 
			items = [x for x in pair]
			print("Rule: " + items[0] + " -> " + items[1])

			#second index of the inner list
			print("Support: " + str(item[1]))

			#third index of the list located at 0th
			#of the third index of the inner list

			print("Confidence: " + str(item[2][0][2]))
			print("Lift: " + str(item[2][0][3]))
			print("Decision by the Bank: " + items[-1])
			print("=====================================")
	
	def hot_encode(self,x): 
		if(x<= 0): 
			return 0
		if(x>= 1): 
			return 1
			
	def runkmc(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(True)
		self.kmcFrame = tk.Frame(self.window, width=1200, height=600)
		self.kmcFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.kmcFrame, bg="pink",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.kmcFrame, bg="pink", fg="white", text ="K-Mode Clustering", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.kmcFrame, width="1200",height = "600").grid(row=1, column=0)
		
	# Machine Learning Technique(s) Menu
	# Decision Tree Classifier On Selected in Menu
	def runDT(self):
		self.destroyFrames()
		self.mlt1Selected.set(True)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.dtFrame = tk.Frame(self.window, width=1200, height=600)
		self.dtFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.dtFrame, bg="pink",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.dtFrame, bg="pink", fg="white", text ="Decision Tree Classifier", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.dtFrame, width="1200",height = "600").grid(row=1, column=0)		
		dtButton = Button(self.dtFrame, text ="Generate Decision Tree", command=self.generateDT).place(x=230,y=250, height=50, width=150)
	# K-Nearest Neighbour On Selected in Menu	
	def runKNN(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(True)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.knnFrame = tk.Frame(self.window, width=1200, height=600)
		self.knnFrame.place(x=0,y=0)
		self.knnValue = IntVar()	
		canvas = tk.Canvas(self.knnFrame, bg="pink",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.knnFrame, bg="pink", fg="white", text ="K-Nearest Neighbour", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.knnFrame, width="1200",height = "600").grid(row=1, column=0)	
		knnSelector = tk.Scale(self.knnFrame, from_=1, to=9, orient=HORIZONTAL, variable=self.knnValue).place(x=250,y=210)		
		knnButton = Button(self.knnFrame, text ="Generate K-NN", command=self.generateKNN).place(x=230,y=250, height=50, width=150)
	# Naive Bayes On Selected in Menu
	def runNB(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(True)
		self.mlt4Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)		
		self.nbFrame = tk.Frame(self.window, width=1200, height=600)
		self.nbFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.nbFrame, bg="pink",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.nbFrame, bg="pink", fg="white", text ="Naive Bayes", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.nbFrame, width="1200",height = "600").grid(row=1, column=0)
		nbButton = Button(self.nbFrame, text ="Generate Naive Bayes", command=self.generateNB).place(x=230,y=250, height=50, width=150)
	# SVM On Selected in Menu
	def runSVM(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(True)	
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.svmFrame = tk.Frame(self.window, width=1200, height=600)
		self.svmFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.svmFrame, bg="pink",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.svmFrame, bg="pink", fg="white", text ="Support Vector Machine", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.svmFrame, width="1200",height = "600").grid(row=1, column=0)
		svmButton = Button(self.svmFrame, text ="Generate SVM", command=self.generateSVM).place(x=230,y=250, height=50, width=150)
	# Destroy Existing Frames when new frame is open
	def destroyFrames(self):
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()	
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()				
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()	
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()	
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()	
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()		
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()	
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()	
		if self.pmSelected == True:
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
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()	
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()	
		if self.b4SmoteSelected == True:
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()	
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()	
		if self.smoteSelected == True:
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()	
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()	
		if self.armSelected == True:
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()	
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()	
		if self.genArmSelected == True:
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()	
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()			
			for widget in self.kmcFrame.winfo_children():
				widget.destroy()
			self.kmcFrame.pack_forget()				
		if self.kmcSelected == True:
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
			for widget in self.predictionFrame.winfo_children():
				widget.destroy()
			self.predictionFrame.pack_forget()	
			for widget in self.b4SmoteFrame.winfo_children():
				widget.destroy()
			self.b4SmoteFrame.pack_forget()	
			for widget in self.smoteFrame.winfo_children():
				widget.destroy()
			self.smoteFrame.pack_forget()
			for widget in self.armFrame.winfo_children():
				widget.destroy()
			self.armFrame.pack_forget()	
			for widget in self.genArmFrame.winfo_children():
				widget.destroy()
			self.genArmFrame.pack_forget()	
			
	# Help Menu
	def aboutUs(self):
		messagebox.showinfo("About Us","TDS3301 - Data Mining Project\nGroup Member:\nOng Shuoh Chwen 1171102212\nYong Wen Kai 1171101664\nLecturer:\nDr. Ting Choo Yee")

	def howToUse(self):
			pass
			messagebox.showinfo("How to use","1. Choose the machine learning technique\n2. Choose the Employment type followed by Type of credit cards, type of properties, "
											 "loan amount and monthly salary\n3. Select the machine learning technique's parameters\n 4. Click predict now")

	# Reset Button On Clicked
	def resetButtonOnClicked(self):
		self.runPM()
		self.destroyFrames()
		
	# Prediction Button On Clicked
	def predictionButtonOnClicked(self):
		clf = svm.SVC(kernel='rbf', gamma='auto', random_state = 1, probability=True)

		#Train the model using the training sets
		clf.fit(self.X_train, self.y_train)

		#Predict the response for test dataset
		self.y_pred = clf.predict(self.X_test)
		
		f1_svm = metrics.f1_score(self.y_test, self.y_pred)
		
		#input=pd.DataFrame(np.array([self.employmentType.get(),self.loanAmount.get(),self.cardType.get(),self.propertyType.get(),self.mthSalary.get()]), columns=['Employment_Type', 'Loan_Amount', 'Credit_Card_types', 'Property_Type', 'Monthly_Salary'])
		#input = input.astype({'Monthly_Salary':'category', 'Employment_Type':'str', 'Decision' : 'str'}, copy=False)
		
		#input = input.apply(lambda x: dictionary[x.name].transform(x))
		#prediction = model.predict(input)
		#prediction = dictionary['Employment'].inverse_transform(prediction)
		
	def saveSelectedValues(self):
		print (self.employmentType.get() + "\n" + 
			   self.cardType.get() + "\n" +
			   self.propertyType.get() + "\n" +		
			   self.loanAmount.get() + "\n" +
			   self.mthSalary.get() + "\n")
	
	# Generate DT Accordingly
	def generateDT(self):
		dt = DecisionTreeClassifier(random_state=1)
		dt = dt.fit(self.X_train, self.y_train)
		self.y_pred = dt.predict(self.X_test)
		
		f1_dt = metrics.f1_score(self.y_test, self.y_pred)		
		f = "Accuracy   :" + str(metrics.accuracy_score(self.y_test, self.y_pred)) + "\nPrecision  :" + str(metrics.precision_score(self.y_test, self.y_pred)) + "\nRecall       :" + str(metrics.recall_score(self.y_test, self.y_pred)) + "\nF1             :" + str(metrics.f1_score(self.y_test, self.y_pred))
		messagebox.showinfo("Decision Tree Classifier",f)
	# Generate KNN Accordingly
	def generateKNN(self):
		k = self.knnValue.get()
		k_range = range(1,10)
		scores = []

		print("Number of K: " + str(k))
		knn = KNeighborsClassifier(n_neighbors = k, weights='uniform')
		knn.fit(self.X_train, self.y_train)
		scores.append(knn.score(self.X_test, self.y_test))
		self.y_pred = knn.predict(self.X_test)
		
		f1_knn = metrics.f1_score(self.y_test, self.y_pred)
			
		f = "No. of K    :" + str(k) + "\nAccuracy   :" + str(metrics.accuracy_score(self.y_test, self.y_pred)) + "\nPrecision  :" + str(metrics.precision_score(self.y_test, self.y_pred)) + "\nRecall       :" + str(metrics.recall_score(self.y_test, self.y_pred)) + "\nF1             :" + str(metrics.f1_score(self.y_test, self.y_pred))
		messagebox.showinfo("K-Nearest Neighbour",f)
		
		plt.figure()
		plt.xlabel('k')
		plt.ylabel('accuracy')
		plt.title('Accuracy by n_neigbors')
		plt.scatter(k_range, scores)
		plt.plot(k_range, scores, color='green', linestyle='dashed', linewidth=1, markersize=5)
	# Generate NB Accordingly
	def generateNB(self):
		nb = GaussianNB()
		nb.fit(self.X_train, self.y_train)
		self.y_pred = nb.predict(self.X_test)
		
		f1_nb = metrics.f1_score(self.y_test, self.y_pred)
		
		f = "Accuracy   :" + str(metrics.accuracy_score(self.y_test, self.y_pred)) + "\nPrecision  :" + str(metrics.precision_score(self.y_test, self.y_pred)) + "\nRecall       :" + str(metrics.recall_score(self.y_test, self.y_pred)) + "\nF1             :" + str(metrics.f1_score(self.y_test, self.y_pred))
		messagebox.showinfo("Naive Bayes",f)
		
	# Generate SVM Accordingly
	def generateSVM(self):
		kernels = ['linear', 'rbf', 'poly']

		for kernel in kernels:
			print("Kernel: " + str(kernel))
			clf = svm.SVC(kernel='rbf', gamma='auto', random_state = 1, probability=True)

			#Train the model using the training sets
			clf.fit(self.X_train, self.y_train)

			#Predict the response for test dataset
			self.y_pred = clf.predict(self.X_test)
			
			if (kernel == 'rbf'):
				f1_svm = metrics.f1_score(self.y_test, self.y_pred)
			f = "Accuracy   :" + str(metrics.accuracy_score(self.y_test, self.y_pred)) + "\nPrecision  :" + str(metrics.precision_score(self.y_test, self.y_pred)) + "\nRecall       :" + str(metrics.recall_score(self.y_test, self.y_pred)) + "\nF1             :" + str(metrics.f1_score(self.y_test, self.y_pred))
			messagebox.showinfo("Support Vector Machine",f)
		
# Displaying the main window
mainWindow = LoanPredictor()
mainWindow.window.mainloop()