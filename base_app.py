"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vec = open("resources/vect.pkl","rb")
news_vectorizer = open("resources/cv_vect (1).pkl","rb")
news_vect = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
tweet_tf = joblib.load(news_vect) # loading your vectorizer from the pkl file
tweet_tv = joblib.load(news_vec) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
predictor1 = joblib.load(open(os.path.join("resources/svc_model (1).pkl"),"rb"))
predictor2 = joblib.load(open(os.path.join("resources/logreg_1_model (2).pkl"),"rb"))
predictor3 = joblib.load(open(os.path.join("resources/model_mmb.pkl"),"rb"))


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Through The Lens Of Twitter")
	st.subheader("Climate change Belief Analysis")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Us", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# models_options = ["Support Vector Classifier", "Logistic Regression", " MultinomialNB"]
	# with st.sidebar:
	# 	model_choice = st.selectbox("Choose Option", models_options)
		
			
	# # Load your .pkl file with the model of your choice + make predictions
	# # Try loading in multiple models to give the user a choice
	# if model_choice == 'Support Vector Classifier':
	# 	predictor = predictor1
			
	# if model_choice == 'Logistic Regression':
	# 	predictor = predictor2
			
	# if model_choice == 'MultinomialNB':
	# 	predictor = predictor3

	# Building out the "about us" page
	if selection == "About Us":
		left_column, right_column = st.columns(2)
		with left_column:
			st.header("The team")
			st.write("##")
			st.write(
				"""
				Daniel Komape - Data Scientist
				Praise Khosa - Data Engineer
				Ulanda Wilcocks - Data Scientist
				Lesego Phaahla - Sale and Marketing Director
				Aphiwe Madela - CEO
				"""
			)

		with right_column:
			st.header("Problement Statement")
			st.write("##")
			st.write(
				"""Climate4Change would like to determine how people perceive climate change and whether or not they believe it is a real threat.
				""")

	

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		vect_text = tweet_cv.transform([tweet_text]).toarray()
		col = st.columns(3)
		col1, col2, col3 = col
		choose1 = col1.button(label="Support Vector Classifier")
		choose2 = col2.button(label="Logistic Regression")
		choose3 = col3.button(label="MultinomialNB")

		if choose1:
			# Transforming user input with vectorizer
			# vect_text = tweet_cv.transform([tweet_text]).toarray()
			prediction = predictor1.predict(vect_text)
			# # Load your .pkl file with the model of your choice + make predictions
			# # Try loading in multiple models to give the user a choice
			# if model_choice == 'Support Vector Classifier':
			# 	predictor = joblib.load(open(os.path.join("resources/svc_model.pkl"),"rb"))
			# 	prediction = predictor.predict(vect_text)
			# elif model_choice == 'Logistic Regression':
			# 	predictor = joblib.load(open(os.path.join("resources/Logreg_1_model (1).pkl"),"rb"))
			# 	prediction = predictor.predict(vect_text)
			# elif model_choice == 'MultinomialNB':
			# 	predictor = joblib.load(open(os.path.join("resources/model_mmb.pkl"),"rb"))
			# 	prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#st.success("Text Categorized as: {}".format(prediction))

		elif choose2:
			prediction = predictor2.predict(vect_text)

			st.success("Text Categorized as: {}".format(prediction))

		elif choose3:
			prediction = predictor3.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
