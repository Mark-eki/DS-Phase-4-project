## Phase 4 Project
# APPLE AND GOOGLE TWITTER SENTIMENT ANALYSIS

## Authors
Neema Naledi,
Henia June,
Morgan Amwai,
Brian Kimathi,
Mark Muriithi.

## 1.Business Understanding
*__1.1 Overview__*
Companies like Apple and Google frequently release new products—from smartphones and software updates to smart home devices. On social platforms such as Twitter (X), users actively express opinions, complaints, and praise about these products.

Analyzing these public sentiments through Natural Language Processing (NLP) provides data-driven insights into customer satisfaction, brand perception, and improvement opportunities—Sentiment analysis with NLP efficiently interprets large amounts of social media data to uncover useful insights that guide marketing, customer experience, and product development strategies.

*__1.2 Stakeholders__:*
- **Apple and Google Product Teams** – to understand user feedback.  
- **Marketing & Brand Managers** – to track brand perception.  
- **Business Analysts & Data Scientists** – to monitor sentiment trends.  

*__1.3 The Problem__:* 
Manually reviewing thousands of tweets about Apple and Google is inefficient and unscalable.  
There's a neeed for an **automated Natural Langauge Processing-based sentiment analysis model** that classifies tweets accurately into categories(positive, negative or neutral)that support data-driven decisions.
 
*__1.4 Objectives__:*
*__1.4.1  Main Objective__:*
To develop an effective sentiment analysis model that is able to automatically classify tweets discussing Apple and Google products as positive, negative or neutral

*__1.4.2 Specific Objectives__*
- Idenitfy patterns in user sentiments.
- Compare the performances of both binary and multiclass algortihms.
- Provide sufficient recommendations to the companies on how to go about the sentiments.


*__1.6 Research Questions__*
1. What is the overall sentiment of Twitter users toward Apple and Google products?
2. Which machine learning approach performs best for classifying tweet sentiment?
3. Can sentiment analysis help identify product or brand perception trends over time?

*__1.7 Success Criteria__:* 
Actionable insights that could inform product or marketing strategies for Apple and Google.

## 2. Data Understanding
**Data set used**:
The dataset judge-1377884607_tweet_product_company.csv was obtained from CrowdFlower via data.world.
It contains tweets mentioning various technology brands, primarily Apple and Google, along with sentiment annotations provided by human raters.
**Key Columns**:

- 'tweet_text':  the text content of the tweet.

- 'is_there_an_emotion_directed_at_a_brand_or_product': target sentiment label (positive, negative, or neutral).

-'emotion_in_tweet_is_directed_at': specifies which company or product the tweet refers to.

**Reasons for data selection**:
1. The dataset provides a large number of real-world tweets, making it ideal for sentiment analysis.

2. It includes manually labeled sentiments, enabling supervised learning.

3. Tweets reference both Apple and Google, allowing comparative brand analysis.

4. The dataset’s size (~9,000 records) is sufficient for model training and evaluation.

**Outliers**: 
No numerical outliers exist since the data is textual.

A few tweets contained non-English text, URLs, emojis, or repeated characters, which were cleaned using regex operations.

Duplicate tweets and those missing tweet_text or sentiment labels were removed to ensure clean data.

Extremely short tweets (1–2 words) were inspected and retained only if meaningful.

**Merged Data**:
The dataset included tweets about multiple brands, but the analysis primarily focused on Apple and Google.

Data for both brands was combined into a single DataFrame (nlp_data), using the column emotion_in_tweet_is_directed_at to distinguish between them.

This merging enabled unified preprocessing, feature extraction, and comparative sentiment analysis between the two companies.


*2.1 Importing Relevant Libraries*

We began by importing the necessary libraries to understand and inspect our data. These libraries assist with data manipulation, visualization, and modeling.

*2.2 Loading the Data*

The dataset was loaded from a Twitter company sentiment dataset and converted into a pandas DataFrame for easier handling and analysis.

*2.3 Initial Exploration and EDA*

An initial exploratory data analysis (EDA) was conducted to understand the dataset’s structure and quality.

Dataset Summary:

Total tweets: 9,093

Total features/columns: 3 (tweet_text, emotion_in_tweet_is_directed_at, and is_there_an_emotion_directed_at_a_brand_or_product)

Missing values: 5,803

Duplicate rows: 22

*Visualizations and Observations*:
*1. Sentiment Distribution Values*
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/4fcc20bb-273b-49e4-a0ab-eea1f7c39efc" />

> Over half of the tweets expressed no emotion toward a brand or product.
> Positive sentiment tweets followed with nearly 300, while negative tweets and “I can’t tell” tweets appeared less frequently.
*2. Counting Brand/Product Mentions*
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/9c02f7e4-fb42-4189-b433-cf153be38431" />

> iPad had the most tweets, indicating strong engagement and sentiment toward that product.
> Other frequently mentioned entities included Apple, Google, and iPhone/iPad apps.
*3.Tweet length Distribution*
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/b5c56a5d-544c-4725-8dd2-f3f0fa860613" />

> The mean tweet length was approximately 105 characters, and a new column was added to the DataFrame to record tweet length.

## 3. Data Preparation
### 3.1 Data Cleaning
1. Rows with missing sentiment labels(missing values) from "is_there_an_emotion_directed_at_a_brand_or_product" were removed to focus on tweers with clear product/brand targerts
2. Remove rows with missing "tweet_text"
3. Remove(drop) all duplicated rows in the "tweet_text" column apart from the first duplicate
4. Manipulate the "tweet text" column: inorder to remain with relevant data and add a "cleaned_tweet" column
 > Tweets contained a lot of noise( such as : URL's, @mentions, hashtag symols, non-ASCII characters, puntuations/symbols, and any extra whitespaces) were all removed
 > Text was converted to lowercase
 > Contraction Expansion: The contractions library expands terms like "can't" to "cannot" to standardize tweet text.

5. Columns were renamed for eadier handling: 'is_there_an_emotion_directed_at_a_brand_or_product': 'sentiment',
    'emotion_in_tweet_is_directed_at': 'product' 

### 3.2 Data Inspection
Here, we view the dataset in visuals after cleaning to see how it looks
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/8cd95b86-c7c8-4950-a7ab-358b4907654e" />

> There is not a lot of difference between the original and the cleaned data but the cleaned data should have fewer rows.

### 3.3 Advanced Insights
1. *Top 15 most common words.*
   <img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/42e6de55-9b7b-42ae-9194-7d031a679735" />
> As seen above,the most common words are; sxsw, lnik and ipad
> while the least common words are; will, popup and this.

2. *Basic summary from original data.*
    <img width="989" height="589" alt="image" src="https://github.com/user-attachments/assets/36073fc4-d820-4316-8d81-3c82535c3390" />

 The cleaned dataset is the saved into a csv file : 'nlp_cleaned_data.csv'

 ## 4.Data Preprocessing
 ### 4.1 Cleaning before preprocessing
 
 1. The cleaned dataset is imported
 2. Columns that were not useful were dropped (ie; 'tweet_text' and 'tweet_length')
 3. Null values were checked:
    > "product" column was dropped as it had too many null values
    > the 1 row in the "cleaned_tweet" column that was missing was also dropped
4. The Distribution in the *"sentiment"* column was checked and the results were as follows:
   sentiment
No emotion toward brand or product    5388
Positive emotion                      2978
Negative emotion                       570
I can't tell                           156

- There is a clear **imbalance** in the sentiment classes
- The imbalance may **affect the model performance as the classifier could become biased toward the majority classes**(Therefore, balancing techniques such as stratified sampling or SMOTE will be considered during the modeling stage.)

### 4.2 Encoding the Sentiment Column

1. First we **merge** *"No emotion toward brand"* and *"I can't tell"* into one **"Neutral"** class
2. Sentiment labels (“positive”, “neutral”, “negative”) were *encoded into numeric values* to make them compatible with machine learning models.
>The mapping was:
                *Negative = 0*
                *Positive = 1*
                *Neutral = 2*
>This facilitated both *binary and multiclass modelling*
3. Then we dropped the "sentiment" column

### 4.3 Preprocessing Tweets
Preprocessing combined multiple steps:
  i.)Tokenization of tweets into words.
 ii.)Stopword removal to exclude uninformative words.
iii.)Lemmatization of the tokenized tweet.
We drop the "cleaned_tweet" column
** The tweets are ready for *vectorization*



## 5. Modelling
Modelling will be done in 2 main parts: 1. Binary Classification
                                                  a) +VE emotion
                                                  b) -VE emotion
                                        2. Multi-class Classification 
                                                       a) +VE emotion
                                                       b) -VE emotion
                                                       c) Neutral emotion

                                                      
 ## 5.1 Binary Modelling
> Data is filtered to include only the +ve and -ve emotion classes
> Vectorization is done
> 80-20 Train-Test split
> Use *SMOTE* *(random_state=42)* to handle the class imbalances: ensuring fair representation of each class before training
Training will be done with multiple models: 1. Logistic Regression Model
                                            2. Multinomial Naives Bayes Model
                                            3. Support Vector Machine(Linear SVC) Model

### 5.1.1 Training a Logistic Regression Model
> It serves as a baseline-model
> Classifier was trained on TF-IDF features(vectorization)
> Perfomance was evaluated using **accuracy, precision, recall and F1-score.**
> Accuracy: 0.847887323943662

### 5.1.2 Training a Multinomial Naive Bayes 
Accuracy score: 0.8535211267605634

### 5.1.3 Training a Support Vector Machine Model
Accuracy score: 0.8915492957746479

#### Confusion Matrix Plots for the 3 Binary Models

<img width="1460" height="424" alt="image" src="https://github.com/user-attachments/assets/359cf2cc-089b-4907-98f5-7e63dadd60a2" />

### 5.1.4 Binary Model Performance Comparison 
From the above matrix confusion matrix, we can conclude that:
1. The linear svm produced the best model out of the three as it was able to predict accurately 89% of the time.
2. The Svm model had a higher false positive (Type 1 error) reading compared to Multinomial Naive Bayes, but a lower false negative(Type 2 error)
3. Logistic regression perfomed the poorest out of the 3 with an accuracy score 84.7 %. Multinomial had an accuracy score of 85.3%.

## 5.2 Multiclass Modelling
In this section, we’ll build models to classify tweets into three sentiment categories:
- Positive emotion (1)
- Negative emotion (0)
- Neutral (2) → merged from “No emotion toward brand or product” and “I can’t tell”.   




 


## Evaluation


## Prerequisites
*Getting started*
1. Fork 
Create a fork.

2. Clone 

- Type: git clone then paste the link below
(you can clone using either *SSH key*  or the *HTTPS*)
[https://github.com/Mark-eki/DS-Phase-4-project.git]

## Key Findings
1. 
2. 
3. 

## Reccomendations 
1. 
2. 
3. 

## Testing
To run the cells press ctrl+shift
You'll need to download the dataset required 
You can get the dataset from:
[https://data.world/crowdflower/brands-and-product-emotions]

## Technologies Used
- Python: Primary programming language
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization
- Jupyter Notebook: Development environment
- Git: Commit and push to remote repository

## Contributions
Contributions to the  are welcome! If you have any suggestions, bug fixes, or additional features you'd like to add to the dashoard, please feel free to submit a pull request or open an issue.

## Support
For questions or support, please contact:
naledineema@gmail.com, heniajune@gmail.com, morganamwai@gmail.com, mark.muriithi@gmail.com, machingabrian@gmail.com


