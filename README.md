# Email-SMS-Spam-Classifier
Email/Spam Classification
During this project, I undertook an machine learning endeavor with the goal of effectively detecting and classifying spam and legitimate SMS messages. For this project, I sourced the SMS Spam Collection Dataset from Kaggle, which provided a carefully curated set of 5,574 SMS messages in English, each labeled as either "ham" (legitimate) or "spam." This dataset served as a valuable resource for my SMS spam research.

The main objective of this project was to develop a robust and accurate spam classification model. To achieve this, I delved into various state-of-the-art machine learning algorithms and employed cutting-edge natural language processing techniques.
The Email SMS Spam Classification Project involves the following steps:

**1. Cleaning the Data:** In this step, the SMS Spam Collection Dataset is thoroughly inspected to identify and handle any missing or irrelevant data. Data cleaning ensures that the dataset is ready for further processing and analysis.

**2. Exploratory Data Analysis (EDA):** EDA is performed to gain insights into the dataset's distribution, statistics, and patterns. This step helps in understanding the characteristics of spam and legitimate messages, enabling better feature selection and engineering.

**3. Text Preprocessing:** Text preprocessing involves converting the raw SMS messages into a format suitable for machine learning. Tasks like tokenization, removing stop words, converting text to lowercase, and applying stemming or lemmatization are performed to standardize the text data.

**4. Model Building:** In this step, various machine learning algorithms, such as Naive Bayes, Support Vector Machines (SVM), K-Nearest Neighbor Algorithm, Decision Tree Classifier, Logistic Regression, Random Forest Classifier, Ada-Boost Classifier, Bagging Classifier, Extra Trees Classifier, Gradient Boosting Classifier and XGB Classifier  are applied to train the spam classification model. Different feature extraction methods, such as Bag-of-Words or TF-IDF, also be utilized. 

**5. Evaluation:** The trained model is evaluated using appropriate performance metrics, such as accuracy, precision. The evaluation ensures that the model can effectively distinguish between spam and legitimate messages.

**6. Improvement:** Based on the evaluation results, adjustments are made to improve the model's performance. This may involve fine-tuning hyperparameters, using different algorithms, or experimenting with different feature engineering techniques. Finally I used Multinomial Naive Bayes with Tf-IDF.

**7. Website Local Hosting:** To provide a user-friendly interface for interacting with the spam classification model, a web application is created. This web application is locally hosted, allowing users to input SMS messages and receive the model's prediction (spam or legitimate).

**8. Deployment:** Once the model and web application are ready, they are deployed using streamlit cloud to make them accessible over the internet. Users can then access the website remotely and utilize the spam classification service. Here is the link of the web app [https://email-sms-spam-classifier.streamlit.app/](url)
