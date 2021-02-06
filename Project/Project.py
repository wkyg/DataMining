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
from sklearn.svm import SVC 
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, neighbors
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from kmodes.kmodes import KModes
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans 
		
class LoanPredictor():

	def __init__(self): 
		# Create Main Display Window
		window = tk.Tk()
		self.dtFrame = Frame()
		self.knnFrame = Frame()
		self.nbFrame = Frame()
		self.svmFrame = Frame()
		self.cmeFrame = Frame()
		self.predictionFrame = Frame()
		self.b4SmoteFrame = Frame()
		self.smoteFrame = Frame()
		self.armFrame = Frame()
		self.genArmFrame = Frame()
		self.kmcFrame = Frame()
		self.employmentType = StringVar()
		self.propertyType = StringVar()
		self.cardType = StringVar()
		self.mthSalary = StringVar()
		self.loanAmount = ''
		self.f1_dt = ''
		self.f1_knn = ''
		self.f1_nb = ''
		self.f1_svm = ''
		self.auc_DT = 0.0
		self.auc_NB = 0.0
		self.auc_KNN = 0.0
		self.auc_SVM = 0.0
		self.prs_DT = 0.0
		self.prs_NB = 0.0
		self.prs_KNN = 0.0
		self.prs_SVM = 0.0	
		self.dictionary = defaultdict(LabelEncoder)
		#pickle_in = open('classifier.pkl', 'rb') 
		#self.classifier = pickle.load(pickle_in)
		self.knnValue = IntVar()
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
		self.mlt5Selected = BooleanVar()		
		technicMenu.add_radiobutton(label = "Decision Tree Classifier", variable = self.mlt1Selected, value=True, command = self.runDT)	
		technicMenu.add_radiobutton(label = "K-Nearest Neighbour", variable = self.mlt2Selected, value=True, command = self.runKNN)
		technicMenu.add_radiobutton(label = "Naive Bayes", variable = self.mlt3Selected, value=True, command = self.runNB)
		technicMenu.add_radiobutton(label = "SVM Kernel", variable = self.mlt4Selected, value=True, command = self.runSVM)
		technicMenu.add_radiobutton(label = "Classification Model Evaluation", variable = self.mlt5Selected, value=True, command = self.runCME)		
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
		
		self.dataPreprocess()
		self.runPM()
	
	# Create Main Display
	def runPM(self):
		self.destroyFrames()
		self.employmentType.set('')
		self.propertyType.set('')
		self.cardType.set('')
		self.mthSalary.set('')
		self.predictionFrame = tk.Frame(self.window, width=1200, height=600)
		self.predictionFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.predictionFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.predictionFrame, bg="SkyBlue3", fg="white", text ="Prediction Model", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.predictionFrame, width="1200",height = "600").grid(row=1, column=0)	
		# Create Sub-Categories
		# SubCategory1 = Employment_Type
		labelSubCat1 = Label(self.predictionFrame, text ="Employment Type", font=('Helvetica', 10), justify=LEFT).place(x=210,y=90)
		empType1 = Radiobutton(self.predictionFrame, text="Employee", variable=self.employmentType, value="0",command=self.saveSelectedValues).place(x=210,y=110)
		empType2 = Radiobutton(self.predictionFrame, text="Employer", variable=self.employmentType, value="1",command=self.saveSelectedValues).place(x=210,y=130)
		empType3 = Radiobutton(self.predictionFrame, text="Fresh Graduate", variable=self.employmentType, value="2", command=self.saveSelectedValues).place(x=210,y=150)
		empType4 = Radiobutton(self.predictionFrame, text="Self Employment", variable=self.employmentType, value="3", command=self.saveSelectedValues).place(x=210,y=170)	
		empType5 = Radiobutton(self.predictionFrame, text="Government", variable=self.employmentType, value="4", command=self.saveSelectedValues).place(x=210,y=190)			
		
		# SubCategory2 = Credit_Card_types
		labelSubCat2 = Label(self.predictionFrame, text ="Type of Credit Cards", font=('Helvetica', 10), justify=LEFT).place(x=420,y=90)
		cardType1 = Radiobutton(self.predictionFrame, text="Normal", variable=self.cardType, value="0",command=self.saveSelectedValues).place(x=420,y=110)
		cardType2 = Radiobutton(self.predictionFrame, text="Gold", variable=self.cardType, value="1",command=self.saveSelectedValues).place(x=420,y=130)
		cardType3 = Radiobutton(self.predictionFrame, text="Platinum", variable=self.cardType, value="2", command=self.saveSelectedValues).place(x=420,y=150)	
		
		# SubCategory3 = Property_Type
		labelSubCat3 = Label(self.predictionFrame, text ="Type of Properties", font=('Helvetica', 10), justify=LEFT).place(x=640,y=90) 
		propertyType1 = Radiobutton(self.predictionFrame, text="Bungalow", variable=self.propertyType, value="0", command=self.saveSelectedValues).place(x=640,y=110)		
		propertyType2 = Radiobutton(self.predictionFrame, text="Condominium", variable=self.propertyType, value="1",command=self.saveSelectedValues).place(x=640,y=130)
		propertyType3 = Radiobutton(self.predictionFrame, text="Flat", variable=self.propertyType, value="2",command=self.saveSelectedValues).place(x=640,y=150)
		propertyType4 = Radiobutton(self.predictionFrame, text="Terrace", variable=self.propertyType, value="3",command=self.saveSelectedValues).place(x=640,y=170)
		
		# SubCategory4 = Monthly_Salary
		labelSubCat4 = Label(self.predictionFrame, text ="Monthly Salary (RM)", font=('Helvetica', 10), justify=LEFT).place(x=860,y=90)
		mthSalary1 = Radiobutton(self.predictionFrame, text="<4,000", variable=self.mthSalary, value="0", command=self.saveSelectedValues).place(x=860,y=110)
		mthSalary2 = Radiobutton(self.predictionFrame, text="4,000 - 7,000", variable=self.mthSalary, value="1", command=self.saveSelectedValues).place(x=860,y=130)
		mthSalary3 = Radiobutton(self.predictionFrame, text="7,000 - 10,000", variable=self.mthSalary, value="2", command=self.saveSelectedValues).place(x=860,y=150)
		mthSalary4 = Radiobutton(self.predictionFrame, text="10,000 - 13,000", variable=self.mthSalary, value="3", command=self.saveSelectedValues).place(x=860,y=170)	
	
		resetBtn = Button(self.predictionFrame, text ="Reset", command=self.resetButtonOnClicked).place(x=450,y=500, height=50, width=150) 	
		predictBtn = Button(self.predictionFrame, text ="Predict Now", command=self.predictionButtonOnClicked).place(x=600,y=500, height=50, width=150)	
		
	# Prediction Button On Clicked
	def predictionButtonOnClicked(self):
		if (self.employmentType.get() == '' or self.cardType.get() == '' or self.propertyType.get() == '' or self.mthSalary.get() == ''):
			messagebox.showwarning("Warning", "Please select all features first!")
		else:
			input=pd.DataFrame(np.array([[self.employmentType.get(), self.cardType.get(), self.propertyType.get(), self.mthSalary.get()]]), columns=['Employment_Type', 'Credit_Card_types', 'Property_Type', 'Monthly_Salary'])
			input = input.astype({'Monthly_Salary':'category'}, copy=False)
			
			input = input.apply(lambda x: self.dictionary[x.name].fit_transform(x))
			prediction = self.model.predict(input)
			prediction = self.dictionary['Decision'].inverse_transform(prediction)
			
			print (prediction[0])
			
			if str(prediction[0]) == 'Reject':
				pred = 'rejected'
				labelPrediction = tk.Label(self.predictionFrame, text ='The bank has ', font=('Helvetica', 15, 'bold'), justify=LEFT).place(x=415,y=450)
				labelPrediction = tk.Label(self.predictionFrame, fg='red', text = str(pred), font=('Helvetica', 15, 'bold'), justify=LEFT).place(x=550,y=450)
				labelPrediction = tk.Label(self.predictionFrame, text =' the loan application.', font=('Helvetica', 15, 'bold'), justify=LEFT).place(x=630,y=450)
			else:
				pred = 'accepted'	
				labelPrediction = tk.Label(self.predictionFrame, text ='The bank has ', font=('Helvetica', 15, 'bold'), justify=LEFT).place(x=415,y=450)
				labelPrediction = tk.Label(self.predictionFrame, fg='green', text = str(pred), font=('Helvetica', 15, 'bold'), justify=LEFT).place(x=550,y=450)
				labelPrediction = tk.Label(self.predictionFrame, text =' the loan application.', font=('Helvetica', 15, 'bold'), justify=LEFT).place(x=645,y=450)	
				
	def saveSelectedValues(self):
		#print (self.employmentType.get() + "\n" + self.cardType.get() + "\n" + self.propertyType.get() + "\n" + self.mthSalary.get() + "\n")
		pass
		
	def dataPreprocess(self):
		# Load Data
		df = pd.read_csv("Bank_CS.csv")
		df1 = df.copy()
		df1
		
		# Duplicate the dataframe
		df1 = df.copy()
		
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
		
		self.df3['Employment_Type']= self.df3['Employment_Type'].map({'Employee':0, 'Employer':1, 'Fresh Graduate':2, 'Self Employment':3, 'Government':4})
		self.df3['Credit_Card_types']= self.df3['Credit_Card_types'].map({'Normal':0, 'Gold':1, 'Platinum':2})
		self.df3['Property_Type']= self.df3['Property_Type'].map({'Bungalow':0, 'Condominium':1, 'Flat':2, 'Terrace':3})
		self.df3['Monthly_Salary']= self.df3['Monthly_Salary'].map({'<4000':0, '4000-7000':1, '7000-10000':2, '10000-13000':3})	
		
		self.df3 = self.df3.apply(lambda x: self.dictionary[x.name].fit_transform(x))
		
		y = self.df3.Decision
		X = self.df3.drop(columns =['Decision'])
		
		smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
		self.X_res, self.y_res = smt.fit_resample(X, y)
		colnames = self.X_res.columns
		
		# Exploratory Data Analysis after SMOTE (Based on Decision)
		self.dfSmote = pd.concat([self.X_res.reset_index(drop=True), self.y_res], axis=1) 
		self.dfSmote = self.dfSmote.apply(lambda x: self.dictionary[x.name].fit_transform(x))		
		self.dfSmote = self.dfSmote.apply(lambda x: self.dictionary[x.name].inverse_transform(x))
		
		# Remove the features with lowest ranking after performing Feature Selection (Boruta and RFE)
		self.X_res.drop(columns=["Number_of_Properties","Loan_Amount"], axis=1, inplace=True)
		
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_res, self.y_res, test_size=0.3,random_state=10)
		self.dfSmote1 = self.dfSmote.apply(lambda x: self.dictionary[x.name].fit_transform(x))
	
		X1 = self.dfSmote1[['Employment_Type', 'Credit_Card_types', 'Property_Type', 'Monthly_Salary']]
		y1 = self.dfSmote1.Decision
		
		# Train the data for prediction model
		X_train_pm, X_test_pm, y_train_pm, y_test_pm = train_test_split(X1, y1, test_size=0.3,random_state=10)
		self.model = GaussianNB()
		self.model.fit(X_train_pm, y_train_pm)
		y_pred_pm = self.model.predict(X_test_pm)
		
	# Visualizing Comparison Charts from Exploratory Data Analysis (Before SMOTE and after SMOTE)
	def runDf2(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.mlt5Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(True)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.b4SmoteFrame = tk.Frame(self.window, width=1200, height=600)
		self.b4SmoteFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.b4SmoteFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.b4SmoteFrame, bg="SkyBlue3", fg="white", text ="EDA (Before SMOTE)", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.b4SmoteFrame, bg='white', width="1200",height = "600").grid(row=1, column=0)

		# which type of employment is likely to have the loan accepted?
		figure = plt.Figure(figsize=(8.5,5), dpi=50)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.b4SmoteFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=0,y=45)
		b = sns.countplot(x=self.df2['Employment_Type'],order=self.df2['Employment_Type'].value_counts().index,hue=self.df2['Decision'], ax=ax)
		for p in b.patches:
			b.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax.set_title('Employment type to have the loan accepted')
						 
		# which type of credit card user is likely to have the loan accepted?
		figure1 = plt.Figure(figsize=(8.5,5), dpi=50)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.b4SmoteFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=410,y=45)
		b1 = sns.countplot(x=self.df2['Credit_Card_types'],order=self.df2['Credit_Card_types'].value_counts().index,hue=self.df2['Decision'], ax=ax1)
		for p in b1.patches:
			b1.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax1.set_title('Type of credit card to have the loan accepted')

		# which type of properties is likely to have the loan accepted?
		figure2 = plt.Figure(figsize=(8.5,5), dpi=50)
		ax2 = figure2.add_subplot(111)
		canvas2 = FigureCanvasTkAgg(figure2,master=self.b4SmoteFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=0,y=300)
		b2 = sns.countplot(x=self.df2['Property_Type'],order=self.df2['Property_Type'].value_counts().index,hue=self.df2['Decision'], ax=ax2)
		for p in b2.patches:
			b2.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax2.set_title('Type of properties to have the loan accepted')		
		
		# what is the monthly salary that is likely to have the loan accepted?
		figure3 = plt.Figure(figsize=(8.5,5), dpi=50)
		ax3 = figure3.add_subplot(111)
		canvas3 = FigureCanvasTkAgg(figure3,master=self.b4SmoteFrame)
		canvas3.draw()
		canvas3.get_tk_widget().place(x=410,y=300)
		b3 = sns.countplot(x=self.df2['Monthly_Salary'],order=self.df2['Monthly_Salary'].value_counts().index,hue=self.df2['Decision'], ax=ax3)
		for p in b3.patches:
			b3.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax3.set_title('Monthly salary to have the loan accepted')	
		
		# what is the decision made by the bank the most frequent?
		figure4 = plt.Figure(figsize=(8,10.5), dpi=50)
		ax4 = figure4.add_subplot(111)
		canvas4 = FigureCanvasTkAgg(figure4,master=self.b4SmoteFrame)
		canvas4.draw()
		canvas4.get_tk_widget().place(x=810,y=45)
		b4 = sns.countplot(x='Decision', data = self.df2, ax=ax4)
		for p in b4.patches:
			b4.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax4.set_title('Most frequent decision made by the bank')		
		
	def runDfSmote(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.mlt5Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(True)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.SmoteFrame = tk.Frame(self.window, width=1200, height=600)
		self.SmoteFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.SmoteFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.SmoteFrame, bg="SkyBlue3", fg="white", text ="EDA (After SMOTE)", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.SmoteFrame, bg='white', width="1200",height = "600").grid(row=1, column=0)
		
		# which type of employment is likely to have the loan accepted?
		figure = plt.Figure(figsize=(8.5,5), dpi=50)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.SmoteFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=0,y=45)
		b = sns.countplot(x=self.dfSmote['Employment_Type'],order=self.dfSmote['Employment_Type'].value_counts().index,hue=self.dfSmote['Decision'], ax=ax)
		for p in b.patches:
			b.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax.set_title('Employment type to have the loan accepted')
						 
		# which type of credit card user is likely to have the loan accepted?
		figure1 = plt.Figure(figsize=(8.5,5), dpi=50)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.SmoteFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=410,y=45)
		b1 = sns.countplot(x=self.dfSmote['Credit_Card_types'],order=self.dfSmote['Credit_Card_types'].value_counts().index,hue=self.dfSmote['Decision'], ax=ax1)
		for p in b1.patches:
			b1.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax1.set_title('Type of credit card to have the loan accepted')

		# which type of properties is likely to have the loan accepted?
		figure2 = plt.Figure(figsize=(8.5,5), dpi=50)
		ax2 = figure2.add_subplot(111)
		canvas2 = FigureCanvasTkAgg(figure2,master=self.SmoteFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=0,y=300)
		b2 = sns.countplot(x=self.dfSmote['Property_Type'],order=self.dfSmote['Property_Type'].value_counts().index,hue=self.dfSmote['Decision'], ax=ax2)
		for p in b2.patches:
			b2.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax2.set_title('Type of properties to have the loan accepted')		
		
		# what is the monthly salary that is likely to have the loan accepted?
		figure3 = plt.Figure(figsize=(8.5,5), dpi=50)
		ax3 = figure3.add_subplot(111)
		canvas3 = FigureCanvasTkAgg(figure3,master=self.SmoteFrame)
		canvas3.draw()
		canvas3.get_tk_widget().place(x=410,y=300)
		b3 = sns.countplot(x=self.dfSmote['Monthly_Salary'],order=self.dfSmote['Monthly_Salary'].value_counts().index,hue=self.dfSmote['Decision'], ax=ax3)
		for p in b3.patches:
			b3.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax3.set_title('Monthly salary to have the loan accepted')	
		
		# what is the decision made by the bank the most frequent?
		figure4 = plt.Figure(figsize=(8,10.5), dpi=50)
		ax4 = figure4.add_subplot(111)
		canvas4 = FigureCanvasTkAgg(figure4,master=self.SmoteFrame)
		canvas4.draw()
		canvas4.get_tk_widget().place(x=810,y=45)
		b4 = sns.countplot(x='Decision', data = self.dfSmote, ax=ax4)
		for p in b4.patches:
			b4.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		ax4.set_title('Most frequent decision made by the bank')	
		
	# Association Rule Mining
	def runARM(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.mlt5Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(True)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.armFrame = tk.Frame(self.window, width=1200, height=600)
		self.armFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.armFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.armFrame, bg="SkyBlue3", fg="white", text ="Association Rule Mining", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.armFrame, width="1200",height = "600").grid(row=1, column=0)
		
		self.df4 = self.dfSmote.copy()
		self.df4 = self.df4.apply(lambda x: self.dictionary[x.name].fit_transform(x))
		self.df4.drop(axis= 1, inplace = True, columns = ['Employment_Type', 'Property_Type', 'Credit_Card_types','Score','Monthly_Salary','State','Decision'])
		
		self.minSuppValue = DoubleVar()
		self.minConfValue = DoubleVar()
		self.minLiftValue = DoubleVar()
		self.maxAnteValue = DoubleVar()
		self.maxConsValue = DoubleVar()
		
		minSuppLabel = tk.Label(self.armFrame, text='Choose the minimum support').place(x=530,y=90)
		minsupp = tk.Scale(self.armFrame, from_=0.0300, to=0.0500, digits = 5, resolution = 0.0001, orient=HORIZONTAL, variable=self.minSuppValue).place(x=570,y=110)
		minConfLabel = tk.Label(self.armFrame, text='Choose the minimum confidence').place(x=530,y=150)		
		minconf = tk.Scale(self.armFrame, from_=0.90, to=0.95, digits = 3, resolution = 0.01, orient=HORIZONTAL, variable=self.minConfValue).place(x=570,y=170)
		minLiftLabel = tk.Label(self.armFrame, text='Choose the minimum lift').place(x=530,y=210)
		minlift = tk.Scale(self.armFrame, from_=0.90, to=1.00, digits = 3, resolution = 0.01, orient=HORIZONTAL, variable=self.minLiftValue).place(x=570,y=230)
		maxAnteLabel = tk.Label(self.armFrame, text='Choose maximum number of antecedent').place(x=510,y=270)		
		maxante = tk.Scale(self.armFrame, from_=2, to=5, orient=HORIZONTAL, variable=self.maxAnteValue).place(x=570,y=290)
		maxConsLabel = tk.Label(self.armFrame, text='Choose maximum number of consequent').place(x=510,y=330)			
		maxcons = tk.Scale(self.armFrame, from_=2, to=5, orient=HORIZONTAL, variable=self.maxConsValue).place(x=570,y=350)
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
		canvas = tk.Canvas(self.genArmFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.genArmFrame, bg="SkyBlue3", fg="white", text ="Association Rule Mining", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.genArmFrame, bg='white', width="1200",height = "600").grid(row=1, column=0)
		resetBtn = Button(self.genArmFrame, text="Reset", command=self.runARM).place(x=200,y=530, height=50, width=150)

		df4_dummy = pd.get_dummies(self.df4)
		
		df4_dummy = df4_dummy.applymap(self.encode_units)
		frequent_itemsets = apriori(df4_dummy, min_support=0.5, use_colnames=True)
		
		records = []		
		for i in range(0, 2350):
			records.append([str(self.df4.values[i,j]) for j in range(0, 7)])
		
		association_rules = apriori(records, min_support=self.minSuppValue.get(), min_confidence=self.minConfValue.get(), min_lift=self.minLiftValue.get(), min_length=2)
		association_rules_list = list(association_rules)
		
		rule = []
		support = []
		confidence = []
		for item in association_rules_list:

			# first index of the inner list
			# Contains base item and add item
			pair = item[0] 
			items = [x for x in pair]
			#print("Rule: " + items[0] + " -> " + items[1])
			rule.append(items[0] + " -> " + items[1])

			#second index of the inner list
			#print("Support: " + str(item[1]))
			support.append(item[1])

			#third index of the list located at 0th
			#of the third index of the inner list

			#print("Confidence: " + str(item[2][0][2]))
			confidence.append(item[2][0][2])
			#print("Lift: " + str(item[2][0][3]))
			#print("=====================================")

		descCanvas = tk.Canvas(self.genArmFrame, bg='white', width="400",height = "440").place(x=80,y=80)	
		labelRules = tk.Label(self.genArmFrame, bg='white',text = "Total Rules Generated: " + str(len(association_rules_list)), font=('Helvetica', 15, 'bold')).place(x=120,y=100)
		labelMinSupp = tk.Label(self.genArmFrame, bg='white', text = "Minimum Support Value: " + str(self.minSuppValue.get()), font=('Helvetica', 12)).place(x=120,y=130)
		labelMinConf = tk.Label(self.genArmFrame, bg='white', text = "Minimum Confidence Value: " + str(self.minConfValue.get()), font=('Helvetica', 12)).place(x=120,y=150)	
		labelMinLift = tk.Label(self.genArmFrame, bg='white', text = "Minimum Lift Value: " + str(self.minLiftValue.get()), font=('Helvetica', 12)).place(x=120,y=170)	
		labelMaxAnte = tk.Label(self.genArmFrame, bg='white', text = "Maximum Antecedent Value: " + str(self.maxAnteValue.get()), font=('Helvetica', 12)).place(x=120,y=190)	
		labelMaxCons = tk.Label(self.genArmFrame, bg='white', text = "Maximum Consequent Value: " + str(self.maxConsValue.get()), font=('Helvetica', 12)).place(x=120,y=210)
			
		labelTopRules = tk.Label(self.genArmFrame, bg='white', text = "Top 3 Rules", font=('Helvetica', 15, 'bold')).place(x=120,y=250)
		labelRule1 = tk.Label(self.genArmFrame, bg='white', text = "Rule 1: " + str(rule[0]), font=('Helvetica', 12)).place(x=120,y=280)
		labelSupp1 = tk.Label(self.genArmFrame, bg='white', text = "Support: " + str(support[0]), font=('Helvetica', 12)).place(x=120,y=300)
		labelConf1 = tk.Label(self.genArmFrame, bg='white', text = "Confidence: " + str(confidence[0]), font=('Helvetica', 12)).place(x=120,y=320)	
		labelRule2 = tk.Label(self.genArmFrame, bg='white', text = "Rule 2: " + str(rule[1]), font=('Helvetica', 12)).place(x=120,y=360)
		labelSupp2 = tk.Label(self.genArmFrame, bg='white', text = "Support: " + str(support[1]), font=('Helvetica', 12)).place(x=120,y=380)
		labelConf2 = tk.Label(self.genArmFrame, bg='white', text = "Confidence: " + str(confidence[1]), font=('Helvetica', 12)).place(x=120,y=400)	
		labelRule3 = tk.Label(self.genArmFrame, bg='white', text = "Rule 3: " + str(rule[2]), font=('Helvetica', 12)).place(x=120,y=440)
		labelSupp3 = tk.Label(self.genArmFrame, bg='white', text = "Support: " + str(support[2]), font=('Helvetica', 12)).place(x=120,y=460)
		labelConf3 = tk.Label(self.genArmFrame, bg='white', text = "Confidence: " + str(confidence[2]), font=('Helvetica', 12)).place(x=120,y=480)			
			
		dfSupp = pd.DataFrame(support, columns=['Support']) 
		dfConf = pd.DataFrame(confidence, columns=['Confidence']) 		
		dfArm = pd.concat([dfSupp.reset_index(drop=True), dfConf], axis=1)
		
		figure = plt.Figure(figsize=(6,6), dpi=90)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.genArmFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=600,y=47)
		ax.scatter(support, confidence, data=dfArm, alpha=0.5, marker="*")
		ax.set_xlabel('Support')
		ax.set_ylabel('Confidence')
	
	def encode_units(self,x): 
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
		self.mlt5Selected.set(False)		
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(True)
		self.kmcFrame = tk.Frame(self.window, width=1200, height=600)
		self.kmcFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.kmcFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.kmcFrame, bg="SkyBlue3", fg="white", text ="K Modes Clustering", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.kmcFrame, bg='white', width="1200",height = "600").grid(row=1, column=0)
		
		self.df5 = pd.concat([self.X_res.reset_index(drop=True), self.y_res], axis=1)
		cost = []
		for num_clusters in list(range(1,5)):
			kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
			kmode.fit_predict(self.df5)
			cost.append(kmode.cost_)

		y = np.array([i for i in range(1,5,1)])
		
		figure = plt.Figure(figsize=(6,6), dpi=47)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.kmcFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=30,y=45)
		ax.plot(y,cost, label='Elbow = 2.0')
		ax.set_xlabel('K')
		ax.set_ylabel('Cost')
		ax.legend()
		ax.set_title('Finding the Elbow for K')
		
		# Chosen cluster = 2, because it is the elbow
		km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
		clusters = km.fit_predict(self.df5)
		self.df5 = self.df5.apply(lambda x: self.dictionary[x.name].fit_transform(x))
		self.df5 = self.df5.apply(lambda x: self.dictionary[x.name].inverse_transform(x))
		
		clusters_df = pd.DataFrame(clusters)
		clusters_df.columns = ['Cluster']
		df5_new = pd.concat([self.df5, clusters_df], axis = 1).reset_index()
		df5_new = df5_new.drop(df5_new.columns[0],axis=1)
		
		
		figure1 = plt.Figure(figsize=(6,6), dpi=47)
		ax1 = figure1.subplots()
		canvas1 = FigureCanvasTkAgg(figure1,master=self.kmcFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=30,y=315)
		ax1.set_title('Number of Clusters = 2')
		b = sns.countplot(x='Cluster', data=df5_new, ax=ax1)
		for p in b.patches:
			b.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
		

		figure2 = plt.Figure(figsize = (27,5), dpi=35)
		ax2 = figure2.subplots()
		canvas2 = FigureCanvasTkAgg(figure2,master=self.kmcFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=300,y=45)
		#ax1.set_title('Number of Clusters = 2')
		b1 = sns.countplot(x=df5_new['Employment_Type'],order=df5_new['Employment_Type'].value_counts().index,hue=df5_new['Cluster'], ax=ax2)
		for p in b1.patches:
			b1.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
			
			
		figure3 = plt.Figure(figsize = (27,5), dpi=35)
		ax3 = figure3.subplots()
		canvas3 = FigureCanvasTkAgg(figure3,master=self.kmcFrame)
		canvas3.draw()
		canvas3.get_tk_widget().place(x=300,y=230)
		#ax1.set_title('Number of Clusters = 2')
		b2 = sns.countplot(x=df5_new['Credit_Card_types'],order=df5_new['Credit_Card_types'].value_counts().index,hue=df5_new['Cluster'], ax=ax3)
		for p in b2.patches:
			b2.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')


		figure4 = plt.Figure(figsize = (27,5), dpi=35)
		ax4 = figure4.subplots()
		canvas4 = FigureCanvasTkAgg(figure4,master=self.kmcFrame)
		canvas4.draw()
		canvas4.get_tk_widget().place(x=300,y=420)
		#ax1.set_title('Number of Clusters = 2')
		b3 = sns.countplot(x=df5_new['Monthly_Salary'],order=df5_new['Monthly_Salary'].value_counts().index,hue=df5_new['Cluster'], ax=ax4)
		for p in b3.patches:
			b3.annotate("%.0f" % p.get_height(), (p.get_x() + 
			p.get_width() / 2., p.get_height()), 
			ha='center', va='center', rotation=0, 
			xytext=(0, 18), textcoords='offset points')
			
	# Machine Learning Technique(s) Menu
	# Decision Tree Classifier On Selected in Menu
	def runDT(self):
		self.destroyFrames()
		self.mlt1Selected.set(True)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.mlt5Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.dtFrame = tk.Frame(self.window, width=1200, height=600)
		self.dtFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.dtFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.dtFrame, bg="SkyBlue3", fg="white", text ="Decision Tree Classifier", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.dtFrame, bg="white", width="1200",height = "600").grid(row=1, column=0)		
		dtButton = Button(self.dtFrame, text ="Generate Decision Tree", command=self.generateDT).place(x=530,y=50, height=50, width=150)
		
	# Generate DT Accordingly
	def generateDT(self):
		model_DT = DecisionTreeClassifier(max_depth=3)
		model_DT.fit(self.X_train, self.y_train)
		self.y_pred = model_DT.predict(self.X_test)
		
		self.f1_dt = metrics.f1_score(self.y_test, self.y_pred)		
		
		bgCanvas = tk.Canvas(self.dtFrame, bg="white", width="400",height = "360").place(x=20,y=130)
		labelTitle = tk.Label(self.dtFrame, bg="white", text ="Performance of Decision Tree", font=('Helvetica', 15, 'bold')).place(x=60,y=160)
		labelAccuracy = tk.Label(self.dtFrame, bg="white", text ="Accuracy: "+ str(metrics.accuracy_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=190)
		labelPrecision = tk.Label(self.dtFrame, bg="white", text ="Precision: "+ str(metrics.precision_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=210)
		labelRecall = tk.Label(self.dtFrame, bg="white", text ="Recall: "+ str(metrics.recall_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=230)	
		labelF1 = tk.Label(self.dtFrame, bg="white", text ="F1 Score: "+ str(metrics.f1_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=250)
		
		confusion_majority=confusion_matrix(self.y_test, self.y_pred)
		labelTitle2 = tk.Label(self.dtFrame, bg="white", text ="Confusion Matrix", font=('Helvetica', 15, 'bold')).place(x=60,y=280)
		labelTN = tk.Label(self.dtFrame, bg="white", text ="Majority TN: "+ str(confusion_majority[0][0]), font=('Helvetica', 12)).place(x=60,y=310)
		labelFP = tk.Label(self.dtFrame, bg="white", text ="Majority FP: "+ str(confusion_majority[0][1]), font=('Helvetica', 12)).place(x=60,y=330)
		labelFN = tk.Label(self.dtFrame, bg="white", text ="Majority FN: "+ str(confusion_majority[1][0]), font=('Helvetica', 12)).place(x=60,y=350)
		labelTP = tk.Label(self.dtFrame, bg="white", text ="Majority TP: "+ str(confusion_majority[1][1]), font=('Helvetica', 12)).place(x=60,y=370)
		
		self.prob_DT = model_DT.predict_proba(self.X_test)
		self.prob_DT = self.prob_DT[:, 1]

		self.auc_DT= roc_auc_score(self.y_test, self.prob_DT)
		self.prs_DT= average_precision_score(self.y_test, self.prob_DT)	
		labelTitle3 = tk.Label(self.dtFrame, bg="white", text ="Curve Scores", font=('Helvetica', 15, 'bold')).place(x=60,y=400)		
		labelAUC = tk.Label(self.dtFrame, bg="white", text ='ROC Curve: %.2f' % self.auc_DT, font=('Helvetica', 12)).place(x=60,y=430)
		labelPRS = tk.Label(self.dtFrame, bg="white", text ='Precision-Recall Curve: %.2f' % self.prs_DT, font=('Helvetica', 12)).place(x=60,y=450)			
		
		figure = plt.Figure(figsize=(6,6), dpi=60)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.dtFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=430,y=130)
		fpr_DT, tpr_DT, thresholds_DT = roc_curve(self.y_test, self.prob_DT)
		ax.plot(fpr_DT, tpr_DT, 'b', label = 'DT')
		ax.plot([0, 1], [0, 1], color='green', linestyle='--')
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('Receiver Operating Characteristic (ROC) Curve')
		ax.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
		figure1 = plt.Figure(figsize=(6,6), dpi=60)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.dtFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=800,y=130)
		prec_DT, rec_DT, threshold_DT = precision_recall_curve(self.y_test, self.prob_DT)
		ax1.plot(prec_DT, rec_DT, color='orange', label='DT') 
		ax1.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
		ax1.set_xlabel('Recall')
		ax1.set_ylabel('Precision')
		ax1.set_title('Precision-Recall Curve')
		ax1.legend(loc = 'lower left')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
	# K-Nearest Neighbour On Selected in Menu	
	def runKNN(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(True)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.mlt5Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.knnFrame = tk.Frame(self.window, width=1200, height=600)
		self.knnFrame.place(x=0,y=0)	
		canvas = tk.Canvas(self.knnFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.knnFrame, bg="SkyBlue3", fg="white", text ="K-Nearest Neighbour", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.knnFrame, bg="white", width="1200",height = "600").grid(row=1, column=0)	
		knnSelector = tk.Scale(self.knnFrame, bg='white', from_=1, to=9, orient=HORIZONTAL, variable=self.knnValue).place(x=455,y=50, height=50, width=150)		
		knnButton = Button(self.knnFrame, text ="Generate K-NN", command=self.generateKNN).place(x=615,y=50, height=50, width=150)
		
	# Generate KNN Accordingly
	def generateKNN(self):
		if self.knnValue.get() == 0:
			k = 3
		else:
			k = self.knnValue.get()
		k_range = range(1,10)
		scores = []
		scores1 = []

		knn = neighbors.KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
		knn.fit(self.X_train, self.y_train)
		scores.append(knn.score(self.X_test, self.y_test))
		
		self.y_pred = knn.predict(self.X_test)
		self.f1_knn = metrics.f1_score(self.y_test, self.y_pred)
		
		self.prob_KNN = knn.predict_proba(self.X_test)
		self.prob_KNN = self.prob_KNN[:, 1]
		
		bgCanvas = tk.Canvas(self.knnFrame, bg="white", width="400",height = "360").place(x=20,y=150)
		labelTitle = tk.Label(self.knnFrame, bg="white", text ="Performance of K-Nearest Neighbor", font=('Helvetica', 15, 'bold')).place(x=60,y=180)
		labelAccuracy = tk.Label(self.knnFrame, bg="white", text ="Accuracy: "+ str(metrics.accuracy_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=210)
		labelPrecision = tk.Label(self.knnFrame, bg="white", text ="Precision: "+ str(metrics.precision_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=230)
		labelRecall = tk.Label(self.knnFrame, bg="white", text ="Recall: "+ str(metrics.recall_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=250)	
		labelF1 = tk.Label(self.knnFrame, bg="white", text ="F1 Score: "+ str(metrics.f1_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=270)
		
		confusion_majority=confusion_matrix(self.y_test, self.y_pred)
		labelTitle2 = tk.Label(self.knnFrame, bg="white", text ="Confusion Matrix", font=('Helvetica', 15, 'bold')).place(x=60,y=300)
		labelTN = tk.Label(self.knnFrame, bg="white", text ="Majority TN: "+ str(confusion_majority[0][0]), font=('Helvetica', 12)).place(x=60,y=330)
		labelFP = tk.Label(self.knnFrame, bg="white", text ="Majority FP: "+ str(confusion_majority[0][1]), font=('Helvetica', 12)).place(x=60,y=350)
		labelFN = tk.Label(self.knnFrame, bg="white", text ="Majority FN: "+ str(confusion_majority[1][0]), font=('Helvetica', 12)).place(x=60,y=370)
		labelTP = tk.Label(self.knnFrame, bg="white", text ="Majority TP: "+ str(confusion_majority[1][1]), font=('Helvetica', 12)).place(x=60,y=390)

		self.auc_KNN= roc_auc_score(self.y_test, self.prob_KNN)
		self.prs_KNN= average_precision_score(self.y_test, self.prob_KNN)
		labelTitle3 = tk.Label(self.knnFrame, bg="white", text ="Curve Scores", font=('Helvetica', 15, 'bold')).place(x=60,y=420)		
		labelAUC = tk.Label(self.knnFrame, bg="white", text ='ROC Curve: %.2f' % self.auc_KNN, font=('Helvetica', 12)).place(x=60,y=450)
		labelPRS = tk.Label(self.knnFrame, bg="white", text ='Precision-Recall Curve: %.2f' % self.prs_KNN, font=('Helvetica', 12)).place(x=60,y=470)		

		figure = plt.Figure(figsize=(6.5,4), dpi=60)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.knnFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=430,y=110)
		fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(self.y_test, self.prob_KNN)
		ax.plot(fpr_KNN, tpr_KNN, 'b', label = 'KNN')
		ax.plot([0, 1], [0, 1], color='green', linestyle='--')
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('Receiver Operating Characteristic (ROC) Curve')
		ax.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
		figure1 = plt.Figure(figsize=(6.5,4), dpi=60)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.knnFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=430,y=350)
		prec_KNN, rec_KNN, threshold_KNN = precision_recall_curve(self.y_test, self.prob_KNN)
		ax1.plot(prec_KNN, rec_KNN, color='orange', label='KNN') 
		ax1.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
		ax1.set_xlabel('Recall')
		ax1.set_ylabel('Precision')
		ax1.set_title('Precision-Recall Curve')
		ax1.legend(loc = 'lower left')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
		for k in k_range:
			knn = neighbors.KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
			knn.fit(self.X_train, self.y_train)
			scores1.append(knn.score(self.X_test, self.y_test))

		figure2 = plt.Figure(figsize=(6,8), dpi=70)
		ax2 = figure2.add_subplot(111)
		canvas2 = FigureCanvasTkAgg(figure2,master=self.knnFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=800,y=60)
		ax2.scatter(k_range, scores1, label="k")
		ax2.plot(k_range, scores1, color='green', linestyle='dashed', linewidth=1, markersize=5)
		ax2.set_xlabel('k')
		ax2.set_ylabel('Accuracy')
		ax2.set_title('Accuracy by n_neigbors')
		ax2.legend(loc = 'upper right')

	# Naive Bayes On Selected in Menu
	def runNB(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(True)
		self.mlt4Selected.set(False)
		self.mlt5Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)		
		self.nbFrame = tk.Frame(self.window, width=1200, height=600)
		self.nbFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.nbFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.nbFrame, bg="SkyBlue3", fg="white", text ="Naive Bayes", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.nbFrame, bg="white", width="1200",height = "600").grid(row=1, column=0)
		nbButton = Button(self.nbFrame, text ="Generate Naive Bayes", command=self.generateNB).place(x=530,y=50, height=50, width=150)
	
	# Generate NB Accordingly
	def generateNB(self):
		nb = GaussianNB()
		nb.fit(self.X_train, self.y_train)
		self.y_pred = nb.predict(self.X_test)
		
		self.f1_nb = metrics.f1_score(self.y_test, self.y_pred)
		
		bgCanvas = tk.Canvas(self.nbFrame, bg="white", width="400",height = "360").place(x=20,y=130)
		labelTitle = tk.Label(self.nbFrame, bg="white", text ="Performance of Naive Bayes", font=('Helvetica', 15, 'bold')).place(x=60,y=160)
		labelAccuracy = tk.Label(self.nbFrame, bg="white", text ="Accuracy: "+ str(metrics.accuracy_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=190)
		labelPrecision = tk.Label(self.nbFrame, bg="white", text ="Precision: "+ str(metrics.precision_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=210)
		labelRecall = tk.Label(self.nbFrame, bg="white", text ="Recall: "+ str(metrics.recall_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=230)	
		labelF1 = tk.Label(self.nbFrame, bg="white", text ="F1 Score: "+ str(metrics.f1_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=250)
		
		confusion_majority=confusion_matrix(self.y_test, self.y_pred)
		labelTitle2 = tk.Label(self.nbFrame, bg="white", text ="Confusion Matrix", font=('Helvetica', 15, 'bold')).place(x=60,y=280)
		labelTN = tk.Label(self.nbFrame, bg="white", text ="Majority TN: "+ str(confusion_majority[0][0]), font=('Helvetica', 12)).place(x=60,y=310)
		labelFP = tk.Label(self.nbFrame, bg="white", text ="Majority FP: "+ str(confusion_majority[0][1]), font=('Helvetica', 12)).place(x=60,y=330)
		labelFN = tk.Label(self.nbFrame, bg="white", text ="Majority FN: "+ str(confusion_majority[1][0]), font=('Helvetica', 12)).place(x=60,y=350)
		labelTP = tk.Label(self.nbFrame, bg="white", text ="Majority TP: "+ str(confusion_majority[1][1]), font=('Helvetica', 12)).place(x=60,y=370)
		
		self.prob_NB = nb.predict_proba(self.X_test)
		self.prob_NB= self.prob_NB[:, 1]

		self.auc_NB= roc_auc_score(self.y_test, self.prob_NB)
		self.prs_NB= average_precision_score(self.y_test, self.prob_NB)
		labelTitle3 = tk.Label(self.nbFrame, bg="white", text ="Curve Scores", font=('Helvetica', 15, 'bold')).place(x=60,y=400)		
		labelAUC = tk.Label(self.nbFrame, bg="white", text ='ROC Curve: %.2f' % self.auc_NB, font=('Helvetica', 12)).place(x=60,y=430)
		labelPRS = tk.Label(self.nbFrame, bg="white", text ='Precision-Recall Curve: %.2f' % self.prs_NB, font=('Helvetica', 12)).place(x=60,y=450)				
		
		figure = plt.Figure(figsize=(6,6), dpi=60)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.nbFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=430,y=130)
		fpr_NB, tpr_NB, thresholds_NB = roc_curve(self.y_test, self.prob_NB)
		ax.plot(fpr_NB, tpr_NB, 'b', label = 'NB')
		ax.plot([0, 1], [0, 1], color='green', linestyle='--')
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('Receiver Operating Characteristic (ROC) Curve')
		ax.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
		figure1 = plt.Figure(figsize=(6,6), dpi=60)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.nbFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=800,y=130)
		prec_NB, rec_NB, threshold_NB = precision_recall_curve(self.y_test, self.prob_NB)
		ax1.plot(prec_NB, rec_NB, color='orange', label='NB') 
		ax1.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
		ax1.set_xlabel('Recall')
		ax1.set_ylabel('Precision')
		ax1.set_title('Precision-Recall Curve')
		ax1.legend(loc = 'lower left')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
					
	# SVM On Selected in Menu
	def runSVM(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(True)	
		self.mlt5Selected.set(False)
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.svmFrame = tk.Frame(self.window, width=1200, height=600)
		self.svmFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.svmFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0, column=0)
		labelMain = tk.Label(self.svmFrame, bg="SkyBlue3", fg="white", text ="Support Vector Machine", font=('Helvetica', 15, 'bold')).grid(row=0, column=0)
		emptyCanvas = tk.Canvas(self.svmFrame, bg="white", width="1200",height = "600").grid(row=1, column=0)
		svmButton = Button(self.svmFrame, text ="Generate SVM", command=self.generateSVM).place(x=530,y=50, height=50, width=150)
		
	# Generate SVM Accordingly
	def generateSVM(self):
		clf = svm.SVC(kernel='rbf', gamma='auto', random_state = 1, probability=True)
		#Train the model using the training sets
		clf.fit(self.X_train, self.y_train)

		#Predict the response for test dataset
		self.y_pred = clf.predict(self.X_test)
		self.f1_svm = metrics.f1_score(self.y_test, self.y_pred)
			
		self.prob_SVM = clf.predict_proba(self.X_test)
		self.prob_SVM= self.prob_SVM[:, 1]

		bgCanvas = tk.Canvas(self.svmFrame, bg="white", width="400",height = "390").place(x=20,y=130)		
		labelKernel = tk.Label(self.svmFrame, bg="white", text ="Kernel: RBF (Radial Basis Function)", font=('Helvetica', 12)).place(x=60,y=150)
		labelTitle = tk.Label(self.svmFrame, bg="white", text ="Performance of SVM", font=('Helvetica', 15, 'bold')).place(x=60,y=180)
		labelAccuracy = tk.Label(self.svmFrame, bg="white", text ="Accuracy: "+ str(metrics.accuracy_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=210)
		labelPrecision = tk.Label(self.svmFrame, bg="white", text ="Precision: "+ str(metrics.precision_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=230)
		labelRecall = tk.Label(self.svmFrame, bg="white", text ="Recall: "+ str(metrics.recall_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=250)	
		labelF1 = tk.Label(self.svmFrame, bg="white", text ="F1 Score: "+ str(metrics.f1_score(self.y_test, self.y_pred)), font=('Helvetica', 12)).place(x=60,y=270)
		
		confusion_majority=confusion_matrix(self.y_test, self.y_pred)
		labelTitle2 = tk.Label(self.svmFrame, bg="white", text ="Confusion Matrix", font=('Helvetica', 15, 'bold')).place(x=60,y=300)
		labelTN = tk.Label(self.svmFrame, bg="white", text ="Majority TN: "+ str(confusion_majority[0][0]), font=('Helvetica', 12)).place(x=60,y=330)
		labelFP = tk.Label(self.svmFrame, bg="white", text ="Majority FP: "+ str(confusion_majority[0][1]), font=('Helvetica', 12)).place(x=60,y=350)
		labelFN = tk.Label(self.svmFrame, bg="white", text ="Majority FN: "+ str(confusion_majority[1][0]), font=('Helvetica', 12)).place(x=60,y=370)
		labelTP = tk.Label(self.svmFrame, bg="white", text ="Majority TP: "+ str(confusion_majority[1][1]), font=('Helvetica', 12)).place(x=60,y=390)
		
		self.auc_SVM= roc_auc_score(self.y_test, self.prob_SVM)
		self.prs_SVM= average_precision_score(self.y_test, self.prob_SVM)
		labelTitle3 = tk.Label(self.svmFrame, bg="white", text ="Curve Scores", font=('Helvetica', 15, 'bold')).place(x=60,y=420)		
		labelAUC = tk.Label(self.svmFrame, bg="white", text ='ROC Curve: %.2f' % self.auc_SVM, font=('Helvetica', 12)).place(x=60,y=450)
		labelPRS = tk.Label(self.svmFrame, bg="white", text ='Precision-Recall Curve: %.2f' % self.prs_SVM, font=('Helvetica', 12)).place(x=60,y=470)		
		
		figure = plt.Figure(figsize=(6.5,4), dpi=60)
		ax = figure.add_subplot(111)
		canvas = FigureCanvasTkAgg(figure,master=self.svmFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=430,y=110)
		fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(self.y_test, self.prob_SVM)
		ax.plot(fpr_SVM, tpr_SVM, 'b', label = 'SVM')
		ax.plot([0, 1], [0, 1], color='green', linestyle='--')
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('Receiver Operating Characteristic (ROC) Curve')
		ax.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
		figure1 = plt.Figure(figsize=(6.5,4), dpi=60)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.svmFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=430,y=350)
		prec_SVM, rec_SVM, threshold_SVM = precision_recall_curve(self.y_test, self.prob_SVM)
		ax1.plot(prec_SVM, rec_SVM, color='orange', label='SVM') 
		ax1.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
		ax1.set_xlabel('Recall')
		ax1.set_ylabel('Precision')
		ax1.set_title('Precision-Recall Curve')
		ax1.legend(loc = 'lower left')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
		X, y = make_blobs(n_samples=2476, centers=2, random_state=0, cluster_std=0.60)
		figure2 = plt.Figure(figsize=(6,8), dpi=70)
		ax2 = figure2.add_subplot(111)
		canvas2 = FigureCanvasTkAgg(figure2,master=self.svmFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=800,y=60)
		ax2.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
		ax2.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c=y, s=50, cmap='autumn')
		
		'''
		xlim = ax2.get_xlim()
		ylim = ax2.get_ylim()

		xx = np.linspace(xlim[0], xlim[1], 30)
		yy = np.linspace(ylim[0], ylim[1], 30)
		YY, XX = np.meshgrid(yy, xx)
		xy = np.vstack([XX.ravel(), YY.ravel()]).T
		Z = clf.decision_function(xy).reshape(XX.shape)
		ax2.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
		ax2.set_xlim(xlim)
		ax2.set_ylim(ylim)
		'''
		
	def runCME(self):
		self.destroyFrames()
		self.mlt1Selected.set(False)
		self.mlt2Selected.set(False)
		self.mlt3Selected.set(False)
		self.mlt4Selected.set(False)
		self.mlt5Selected.set(True)		
		self.pmSelected.set(False)
		self.b4SmoteSelected.set(False)
		self.smoteSelected.set(False)
		self.armSelected.set(False)
		self.genArmSelected.set(False)
		self.kmcSelected.set(False)
		self.cmeFrame = tk.Frame(self.window, width=1200, height=600)
		self.cmeFrame.place(x=0,y=0)
		canvas = tk.Canvas(self.cmeFrame, bg="SkyBlue3",width="1200",height = "40").grid(row=0,column=0)
		labelMain = tk.Label(self.cmeFrame, bg="SkyBlue3", fg="white", text ="Classification Model Evaluation", font=('Helvetica', 15, 'bold')).grid(row=0,column=0)
		emptyCanvas = tk.Canvas(self.cmeFrame, bg="white", width="1200",height = "600").grid(row=1, column=0)
		
		# DT
		model_DT = DecisionTreeClassifier(max_depth=3)
		model_DT.fit(self.X_train, self.y_train)
		self.y_pred = model_DT.predict(self.X_test)
		
		self.f1_dt = metrics.f1_score(self.y_test, self.y_pred)		
		
		self.prob_DT = model_DT.predict_proba(self.X_test)
		self.prob_DT = self.prob_DT[:, 1]
		
		self.auc_DT= roc_auc_score(self.y_test, self.prob_DT)
		self.prs_DT= average_precision_score(self.y_test, self.prob_DT)
		
		# KNN
		if self.knnValue.get() == 0:
			k = 3
		else:
			k = self.knnValue.get()
		knn = neighbors.KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
		knn.fit(self.X_train, self.y_train)
		
		self.y_pred = knn.predict(self.X_test)
		self.f1_knn = metrics.f1_score(self.y_test, self.y_pred)
		
		self.prob_KNN = knn.predict_proba(self.X_test)
		self.prob_KNN = self.prob_KNN[:, 1]
		
		self.auc_KNN= roc_auc_score(self.y_test, self.prob_KNN)
		self.prs_KNN= average_precision_score(self.y_test, self.prob_KNN)
		
		# NB
		nb = GaussianNB()
		nb.fit(self.X_train, self.y_train)
		self.y_pred = nb.predict(self.X_test)
		
		self.f1_nb = metrics.f1_score(self.y_test, self.y_pred)
		
		self.prob_NB = nb.predict_proba(self.X_test)
		self.prob_NB= self.prob_NB[:, 1]
		
		self.auc_NB= roc_auc_score(self.y_test, self.prob_NB)
		self.prs_NB= average_precision_score(self.y_test, self.prob_NB)
		
		# SVM
		clf = svm.SVC(kernel='rbf', gamma='auto', random_state = 1, probability=True)
		#Train the model using the training sets
		clf.fit(self.X_train, self.y_train)

		#Predict the response for test dataset
		self.y_pred = clf.predict(self.X_test)
		self.f1_svm = metrics.f1_score(self.y_test, self.y_pred)
			
		self.prob_SVM = clf.predict_proba(self.X_test)
		self.prob_SVM= self.prob_SVM[:, 1]
		
		self.auc_SVM= roc_auc_score(self.y_test, self.prob_SVM)
		self.prs_SVM= average_precision_score(self.y_test, self.prob_SVM)
		
		scores = {'Decision Tree': self.f1_dt, 'KNN': self.f1_knn, 'Naive Bayes': self.f1_nb, 'SVM': self.f1_svm}
		model = list(scores.keys())
		score = list(scores.values())
		
		figure = plt.Figure(figsize=(7,6), dpi=90)
		ax = figure.add_subplot(111)
		ax.bar(model, score, width=0.5)
		ax.set_title("F1-score of classification models")
		canvas = FigureCanvasTkAgg(figure,master=self.cmeFrame)
		canvas.draw()
		canvas.get_tk_widget().place(x=550,y=50)
		
		for x,y in zip(model,score):

			label = "{:.2f}".format(y)

			ax.annotate(label,
						 (x,y), 
						 textcoords="offset points", 
						 xytext=(0,5),
						 ha='center')
						 
		
		figure1 = plt.Figure(figsize=(10,4.5), dpi=60)
		ax1 = figure1.add_subplot(111)
		canvas1 = FigureCanvasTkAgg(figure1,master=self.cmeFrame)
		canvas1.draw()
		canvas1.get_tk_widget().place(x=0,y=45)
		fpr_NB, tpr_NB, thresholds_NB = roc_curve(self.y_test, self.prob_NB)
		fpr_DT, tpr_DT, thresholds_DT = roc_curve(self.y_test, self.prob_DT)
		fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(self.y_test, self.prob_KNN)
		fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(self.y_test, self.prob_SVM)		
		ax1.plot(fpr_NB, tpr_NB, color='blue', label = 'NB: %.2f' % self.auc_NB)
		ax1.plot(fpr_DT, tpr_DT, color='yellow', label = 'DT: %.2f' % self.auc_DT)
		ax1.plot(fpr_KNN, tpr_KNN, color='red', label = 'KNN: %.2f' % self.auc_KNN)	
		ax1.plot(fpr_SVM, tpr_SVM, color='green', label = 'SVM: %.2f' % self.auc_SVM)		
		ax1.plot([0, 1], [0, 1], color='black', linestyle='--')
		ax1.set_xlabel('False Positive Rate')
		ax1.set_ylabel('True Positive Rate')
		ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
		ax1.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		
		
		figure2 = plt.Figure(figsize=(10,4.5), dpi=60)
		ax2 = figure2.add_subplot(111)
		canvas2 = FigureCanvasTkAgg(figure2,master=self.cmeFrame)
		canvas2.draw()
		canvas2.get_tk_widget().place(x=0,y=310)
		prec_NB, rec_NB, threshold_NB = precision_recall_curve(self.y_test, self.prob_NB)
		prec_DT, rec_DT, threshold_DT = precision_recall_curve(self.y_test, self.prob_DT)
		prec_KNN, rec_KNN, threshold_KNN = precision_recall_curve(self.y_test, self.prob_KNN)
		prec_SVM, rec_SVM, threshold_SVM = precision_recall_curve(self.y_test, self.prob_SVM)		
		ax2.plot(prec_NB, rec_NB, color='orange', label='NB: %.2f' % self.prs_NB) 
		ax2.plot(prec_DT, rec_DT, color='green', label='DT: %.2f' % self.prs_DT)
		ax2.plot(prec_KNN, rec_KNN, color='red', label='KNN: %.2f' % self.prs_KNN)	
		ax2.plot(prec_SVM, rec_SVM, color='blue', label='SVM: %.2f' % self.prs_SVM)		
		ax2.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
		ax2.set_xlabel('Recall')
		ax2.set_ylabel('Precision')
		ax2.set_title('Precision-Recall Curve')
		ax2.legend(loc = 'upper left')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])

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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()				
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()				
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
		if self.mlt5Selected == True:
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
			for widget in self.svmFrame.winfo_children():
				widget.destroy()
			self.svmFrame.pack_forget()				
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
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
			for widget in self.cmeFrame.winfo_children():
				widget.destroy()
			self.cmeFrame.pack_forget()	
			
	# Help Menu
	def aboutUs(self):
		messagebox.showinfo("About Us","TDS3301 - Data Mining Project\nGroup Member:\nOng Shuoh Chwen 1171102212\nYong Wen Kai 1171101664\nLecturer:\nDr. Ting Choo Yee")

	def howToUse(self):
			pass
			messagebox.showinfo("How to use",
			"Association Rule Mining: Discover all unique rules.\n"
			"EDA: Data preprocess before and after applying SMOTE.\n"
			"Machine Learning Techniques: Run MLT to find the best fit prediction model.\n"
			"Clustering: Cluster the data into different proportions.\n"
			"Prediction Model: Predict the loan application.")
			

	# Reset Button On Clicked
	def resetButtonOnClicked(self):
		self.runPM()
		self.employmentType.set('')
		self.propertyType.set('')
		self.cardType.set('')
		self.mthSalary.set('')
		self.destroyFrames()
		
# Displaying the main window
mainWindow = LoanPredictor()
mainWindow.window.mainloop()