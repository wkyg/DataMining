from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import missingno as msno
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
		helpMenu.add_command(label = "How to Use", command = self.howToUse)			
		menubar.add_cascade(label = "Help", menu = helpMenu)
		
		self.mainDisplay()
		self.dataPreprocess()
	
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
		
	def dataPreprocess(self):
		# Load Data
		df = pd.read_csv("Bank_CS.csv")
		df1 = df.copy()
		df1
		print (df1.head())
		
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
		
		# Show all Data Types of Df1
		print(df1.dtypes)	
		
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
		df2 = df1.copy()
		
		# Change Loan_Amount to Categorical DType
		df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(100000.0,300000.0), '100000-300000', df2['Loan_Amount'])
		df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(300000.0,500000.0), '300000-500000', df2['Loan_Amount'])
		df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(500000.0,700000.0), '500000-700000', df2['Loan_Amount'])
		df2['Loan_Amount'] = np.where(df1['Loan_Amount'].between(700000.0,900000.0), '700000-900000', df2['Loan_Amount'])
		df2['Loan_Amount'] = df2['Loan_Amount'].astype("category")

		# Change Loan_Amount to Categorical DType
		df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'] < 4000.0, '<4000', df2['Monthly_Salary'])
		df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'].between(4000.0,7000.0), '4000-7000', df2['Monthly_Salary'])
		df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'].between(7000.0,10000.0), '7000-10000', df2['Monthly_Salary'])
		df2['Monthly_Salary'] = np.where(df1['Monthly_Salary'].between(10000.0,13000.0), '10000-13000', df2['Monthly_Salary'])
		df2['Monthly_Salary'] = df2['Monthly_Salary'].astype("category")
		
		#Applying SMOTE based on Decision
		df3 = df2.copy()

		dictionary = defaultdict(LabelEncoder)
		df3 = df2.apply(lambda x: dictionary[x.name].fit_transform(x))

		y = df3.Decision
		X = df3.drop(columns =['Decision'])
		
		smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
		self.X_res, self.y_res = smt.fit_resample(X, y)
		colnames = self.X_res.columns
		
		# Exploratory Data Analysis after SMOTE (Based on Decision)
		dfSmote = pd.concat([self.X_res.reset_index(drop=True), self.y_res], axis=1) 
		dfSmote = dfSmote.apply(lambda x: dictionary[x.name].inverse_transform(x))
		print(dfSmote.head())
		
		# Remove the features with lowest ranking after performing Feature Selection (Boruta and RFE)
		self.X_res.drop(columns=["Number_of_Properties","Loan_Amount"], axis=1, inplace=True)
		'''
		# Association Rule Mining
		df4 = df3.copy()
		df4.drop(axis= 1, inplace = True, columns = ['Employment_Type', 'Credit_Card_types','Property_Type','Monthly_Salary','Loan_Amount'])
		df4 = df4.apply(lambda x: dictionary[x.name].inverse_transform(x))
		df4_dummy = pd.get_dummies(df4)
		
		frequent = apriori(df4_dummy, min_support=0.5, use_colnames=True)
		rules = association_rules(frequent, metric="lift", min_threshold=1.0)
		rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
		rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
		rules[(rules['confidence'] > 0.5) & (rules['antecedent_len'] <= 2) & (rules['consequent_len'] <= 2)].nlargest(10, 'lift')
		'''
		# Machine Learning Techiniques
		# Support Vector Machine (SVM)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_res, self.y_res, test_size=0.3,random_state=1)
		
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
		dtButton = Button(self.dtFrame, text ="Generate Decision Tree Classifier", command=self.generateDT).grid(row=1, column=0)
	# K-Nearest Neighbour On Selected in Menu	
	def runKNN(self):
		self.checkMLT()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(True)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.knnFrame = tk.Frame(self.window, width=600, height=50)
		self.knnFrame.place(x=210,y=400)
		self.knnValue = IntVar()	
		knnSelector = tk.Scale(self.knnFrame, from_=1, to=9, orient=HORIZONTAL, variable=self.knnValue).grid(row=0, column=0)
		knnButton = Button(self.knnFrame, text ="Generate K-Nearest Neighbour", command=self.generateKNN).grid(row=1, column=0)
	# Naive Bayes On Selected in Menu
	def runNB(self):
		self.checkMLT()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(True)
		self.mlt4Selected.set(False)	
		self.nbFrame = tk.Frame(self.window, width=600, height=50)
		self.nbFrame.place(x=210,y=400)
		nbButton = Button(self.nbFrame, text ="Generate Naive Bayes", command=self.generateNB).grid(row=1, column=0)
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
		svmButton = Button(self.svmFrame, text ="Generate SVM", command=self.generateSVM).grid(row=3, column=0)
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
			messagebox.showinfo("How to use","1. Choose the machine learning technique\n2. Choose the Employment type followed by Type of credit cards, type of properties, "
											 "loan amount and monthly salary\n3. Select the machine learning technique's parameters\n 4. Click predict now")
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
		print("Predicted Value")
		clf = svm.SVC(kernel='rbf', gamma='auto', random_state = 1, probability=True)

		#Train the model using the training sets
		clf.fit(self.X_train, self.y_train)

		#Predict the response for test dataset
		self.y_pred = clf.predict(self.X_test)
		
		f1_svm = metrics.f1_score(self.y_test, self.y_pred)
		
		input=pd.DataFrame(np.array([self.employmentType.get(),self.loanAmount.get(),self.cardType.get(),self.propertyType.get(),self.mthSalary.get()]), columns=['Employment_Type', 'Loan_Amount', 'Credit_Card_types', 'Property_Type', 'Monthly_Salary'])
		input = input.astype({'Monthly_Salary':'category', 'Employment_Type':'str', 'Decision' : 'str'}, copy=False)
		
		input = input.apply(lambda x: dictionary[x.name].transform(x))
		prediction = model.predict(input)
		prediction = dictionary['Employment'].inverse_transform(prediction)
		
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

		print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
		print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
		print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
		print("F1:",metrics.f1_score(self.y_test, self.y_pred))			

	# Generate KNN Accordingly
	def generateKNN(self):
		k = self.knnValue.get()
		scores = []

		print("Number of K: " + str(k))
		knn = KNeighborsClassifier(n_neighbors = k, weights='uniform')
		knn.fit(self.X_train, self.y_train)
		scores.append(knn.score(self.X_test, self.y_test))
		self.y_pred = knn.predict(self.X_test)
		
		f1_knn = metrics.f1_score(self.y_test, self.y_pred)
			
		print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
		print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
		print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
		print("F1:",metrics.f1_score(self.y_test, self.y_pred))
		print("\n")
	
	# Generate NB Accordingly
	def generateNB(self):
		nb = GaussianNB()
		nb.fit(self.X_train, self.y_train)
		self.y_pred = nb.predict(self.X_test)
		
		f1_nb = metrics.f1_score(self.y_test, self.y_pred)

		print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
		print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
		print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
		print("F1:",metrics.f1_score(self.y_test, self.y_pred))
		
	# Generate SVM Accordingly
	def SVMRbf(self): 
		print("Kernel: Rbf")
		clf = svm.SVC(kernel='rbf', gamma='auto', random_state = 1, probability=True)

		#Train the model using the training sets
		clf.fit(self.X_train, self.y_train)

		#Predict the response for test dataset
		self.y_pred = clf.predict(self.X_test)
		
		f1_svm = metrics.f1_score(self.y_test, self.y_pred)

		print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
		print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
		print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
		print("F1:",metrics.f1_score(self.y_test, self.y_pred))
		print("\n")
	def SVMLinear(self): 
		print("Kernel: Linear")
		clf = svm.SVC(kernel='linear', gamma='auto', random_state = 1, probability=True)

		#Train the model using the training sets
		clf.fit(self.X_train, self.y_train)

		#Predict the response for test dataset
		self.y_pred = clf.predict(self.X_test)

		print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
		print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
		print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
		print("F1:",metrics.f1_score(self.y_test, self.y_pred))
		print("\n")
	def SVMPoly(self): 
		print("Kernel: Polynomial")
		clf = svm.SVC(kernel='poly', gamma='auto', random_state = 1, probability=True)

		#Train the model using the training sets
		clf.fit(self.X_train, self.y_train)

		#Predict the response for test dataset
		self.y_pred = clf.predict(self.X_test)

		print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
		print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
		print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
		print("F1:",metrics.f1_score(self.y_test, self.y_pred))
		print("\n")
	def generateSVM(self):
		pass
		
# Displaying the main window
mainWindow = LoanPredictor()
mainWindow.window.mainloop()