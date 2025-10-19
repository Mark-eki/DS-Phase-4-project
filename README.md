## Phase 4 Project
# APPLE AND GOOGLE TWITTER SENTIMENT ANALYSIS


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

Observations*:
*1. Sentiment Distribution Values*
> Over half of the tweets expressed no emotion toward a brand or product.
> Positive sentiment tweets followed with nearly 300, while negative tweets and “I can’t tell” tweets appeared less frequently.
*2. Counting Brand/Product Mentions*
> iPad had the most tweets, indicating strong engagement and sentiment toward that product.
> Other frequently mentioned entities included Apple, Google, and iPhone/iPad apps.
*3.Tweet length Distribution*
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
> Ffrom hte visualization as seen in the notebook;There is not a lot of difference between the original and the cleaned data but the cleaned data should have fewer rows.

### 3.3 Advanced Insights
1. *Top 15 most common words.*
   
> The most common words are; sxsw, lnik and ipad
> while the least common words are; will, popup and this.

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

### 5.2.1 Training a Multiclass Logistic Regression Model
Multiclass Logistic Regression Performance:

              precision    recall  f1-score   support

           0       0.37      0.49      0.42       114
           1       0.59      0.60      0.60       596
           2       0.77      0.73      0.75      1109

    accuracy                           0.68      1819
   macro avg       0.58      0.61      0.59      1819
weighted avg       0.68      0.68      0.68      1819

Accuracy: 0.6750962067069819


### 5.2.2 Training a Multiclass Multinomial Naive Bayes Model
Multiclass Naive Bayes Performance:
              precision    recall  f1-score   support

           0       0.29      0.57      0.38       114
           1       0.53      0.71      0.61       596
           2       0.81      0.58      0.67      1109

    accuracy                           0.62      1819
   macro avg       0.54      0.62      0.55      1819
weighted avg       0.69      0.62      0.63      1819

Accuracy: 0.6206706981858163


### 5.2.3 Training a Support Vector Machine Model
Multiclss SVM Performance:

              precision    recall  f1-score   support

           0       0.45      0.47      0.46       114
           1       0.60      0.63      0.61       596
           2       0.77      0.75      0.76      1109

    accuracy                           0.69      1819
   macro avg       0.61      0.62      0.61      1819
weighted avg       0.69      0.69      0.69      1819

Accuracy: 0.6915887850467289


### 5.2.4 Multiclass Model Performance Comparison
According to the above, we can see that:
1. The SVM model still did well with an acccurcy score of 69.1 % able to predict the targets more accurately.
2. Here, with the inclusion of a third target, the logistic regression model did better with an accuracy score of 67.5 % compare to Multinomial Naive Bayes which had 62% accuracy.
3. SVM model and logistics had high false positives with Multinomial Naive Bayes has high false negatives.


### 5.2.5 Multiclass Hyperparameter Tuning on SVM Model
GridSearchCV was applied to optimize the C parameter for SVM.
The tuning process slightly improved model performance, confirming the chosen hyperparameters were effective.
Best params: {'C': 10}
Best Accuracy: 0.8523111612175873

Test Set Accuracy: 0.6827927432655305
              precision    recall  f1-score   support

           0       0.45      0.41      0.43       114
           1       0.58      0.61      0.60       596
           2       0.76      0.75      0.75      1109

    accuracy                           0.68      1819
   macro avg       0.60      0.59      0.59      1819
weighted avg       0.68      0.68      0.68      1819


### 5.2.6 Hyperparameter Tuning Performance Review
After tuning our Support Vector Machine model, the training accuracy is 85% with testing accuracy still 69%.
Most notable is Class 1 target errors are heavily biased towards being misclassified as Class 2 213 times and Class 2 target errors are heavily biased towards being misclassified as Class 1 240 times.


 


## 6.0 Evaluation, Recommendations, and Conclusion
### 6.1 Overview

This section evaluates the performance of the sentiment classification models and provides recommendations for improvement and future work.

The project involved:

Binary sentiment classification (positive vs. negative)

Multi-class sentiment classification (negative, neutral, positive)

Feature engineering: TF-IDF vectorization for text representation

Balancing technique: SMOTE to address class imbalance

Models tested: Logistic Regression, Naive Bayes, LinearSVC

Hyperparameter tuning: GridSearchCV for optimization

TF-IDF effectively represented textual data numerically, while SMOTE improved model fairness by synthetically balancing minority sentiment classes.

### 6.2 Evaluation

Both binary and multi-class classification models were implemented to analyze sentiment from Apple and Google tweets.
TF-IDF ensured strong feature representation, and SMOTE improved recall for underrepresented sentiments.

#### Model Performance Summary

Model	Accuracy	Precision	Recall	F1-score	Notes
Logistic Regression	0.847	0.84	0.84	0.84	Baseline
Naive Bayes	0.853	0.85	0.85	0.85	Baseline
LinearSVC (Tuned)	0.690	0.70	0.69	0.69	Tuned

#### Key Findings

Best Performing Model: Tuned LinearSVC achieved the most balanced performance across all metrics.

Impact of SMOTE: Improved recall for minority classes, especially negative sentiments.

Common Misclassifications: Neutral tweets were often confused with positive ones due to subtle tone or sarcasm.

#### Confusion Matrix Insights

False Positives: Neutral tweets predicted as positive.

False Negatives: Positive tweets predicted as neutral.

Neutral sentiments proved the hardest to classify, largely because of linguistic ambiguity and mixed emotions typical in social media discourse.

### 6.3 Recommendations

Data Enrichment:
Expand the dataset with recent and diverse tweets to improve generalization across slang, emojis, and evolving language patterns.

Advanced Text Representations:
Implement contextual word embeddings (e.g., Word2Vec, GloVe) or transformer-based models (e.g., BERT, RoBERTa) for deeper semantic understanding.

Contextual Awareness:
Use models capable of detecting sarcasm and nuanced emotion through contextual learning.

Real-Time Dashboard:
Integrate the tuned LinearSVC into a live monitoring system for marketing and customer engagement insights.

Aspect-Based Sentiment Analysis:
Extend the analysis to evaluate sentiment toward specific product features such as camera quality, battery life, or performance.

### 6.4 Summary of Modelling Results

Binary Classification:
Naive Bayes achieved the highest accuracy (85.3%), outperforming Logistic Regression (84.7%) with strong generalization between positive and negative sentiments.

Multi-Class Classification:
The tuned LinearSVC achieved 85% training and 69% testing accuracy, showing balanced precision and recall.
Logistic Regression and Naive Bayes followed at 67.5% and 62%, respectively.

Trends:
Neutral tweets caused the most misclassifications due to ambiguous tone and informal language such as slang and emojis.

### 6.5 Limitations

Minority Class Prediction:
Despite SMOTE, predicting neutral sentiment remained challenging due to overlapping linguistic patterns.

Model Limitations:
TF-IDF combined with classical ML models (Logistic Regression, Naive Bayes, LinearSVC) lacks semantic depth, limiting contextual understanding.

Linguistic Complexity:
Elements like sarcasm, abbreviations, and emojis reduce interpretability for traditional ML approaches.

### 6.6 Conclusion

Binary Classification:
Naive Bayes delivered the highest accuracy (85.3%), slightly outperforming Logistic Regression (84.7%).

Multi-Class Classification:
Tuned LinearSVC achieved 85% training and 69% testing accuracy, balancing precision and recall effectively.

#### Key Takeaways

Model Strengths: Naive Bayes excelled in binary classification; LinearSVC performed best in multi-class tasks.

Accuracy Trends: Positive and negative sentiments were well captured, while neutral sentiments remained ambiguous.

Objective Fulfillment: Models successfully automated sentiment classification, generating data-driven insights into brand perception.

#### Future Work:

Integrate contextual embeddings or transformer architectures for deeper semantic learning.

Deploy models in real-time systems for continuous sentiment tracking.

Expand to aspect-based sentiment analysis for targeted business insights.

### Project Collaborators
1. Neema Naledi (naledineema@gmail.com)
2. Henia June (heniajune@gmail.com)
3. Morgan Amwai (morganamwai@gmail.com)
4. Brian Kimathi (machingabrian@gmail.com)
5. Mark Muriithi (mark.muriithi@gmail.com)

### Navigating the Repository

1. Jupyter Notebook: ([nlp.ipynb](https://github.com/Mark-eki/DS-Phase-4-project/blob/Morgan/nlp.ipynb)
2. Presentation slides PDF:
3. Data Report:
4. Dataset:[judge-1377884607_tweet_product_company.csv](https://data.world/crowdflower/brands-and-product-emotions)
5. README.md: Project Overview
6. .gitignore: Specifies files to ignore in version control



## Prerequisites
*Getting started*
1. Fork 
- Create a fork.
2. Clone 
- Type: git clone then paste the link below
(you can clone using either *SSH key*  or the *HTTPS*)
[https://github.com/Mark-eki/DS-Phase-4-project.git]

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


