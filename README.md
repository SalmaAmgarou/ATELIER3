## Predicting Coding Answer Scores with NLP and Machine Learning

This repository contains a Jupyter Notebook demonstrating how to predict the scores of coding answers using natural language processing (NLP) and machine learning. 

### Approach

The approach involves:

1. **Data Loading and Preprocessing:** Loading a CSV dataset containing coding answers, scores, and correctness labels. The text data is then preprocessed using techniques like:
   * **Tokenization:** Breaking down text into individual words.
   * **Stop Word Removal:** Removing common words that have little semantic meaning (e.g., "the", "a", "is").
   * **Lemmatization:** Reducing words to their base form (e.g., "running" -> "run").
   * **Discretization:**  Creating categorical features from numerical data, such as answer length or word count.

2. **Data Encoding:** Encoding the preprocessed text into numerical representations suitable for machine learning. The notebook utilizes three encoding methods:
   * **Word2Vec:**  Creating vector representations of words based on their context within the text.
   * **Bag-of-Words:**  Representing each answer as a vector of word counts.
   * **TF-IDF:**  Weighting words based on their frequency and importance within the corpus.

3. **Model Training and Evaluation:** Training four regression models on the encoded data:
   * **Support Vector Regression (SVR):** A powerful algorithm for non-linear regression.
   * **Naive Bayes:** A probabilistic model that assumes independence between features.
   * **Linear Regression:** A simple linear model that seeks a linear relationship between features and target.
   * **Decision Tree Regressor:** A tree-based model that splits data recursively based on features.

   The models are evaluated using standard regression metrics:
   * **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values.
   * **Root Mean Squared Error (RMSE):** The square root of MSE, providing a more interpretable error measure.
   * **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values.
   * **R-squared (R²):** Represents the proportion of variance explained by the model.
   * **Explained Variance Score (EVS):**  Measures the amount of variance in the target variable explained by the model.

4. **Model Selection and Interpretation:**  Comparing the performance of the models based on the evaluation metrics and selecting the best-performing one.  The analysis aims to identify the model that provides the most accurate predictions and offers insights into the factors influencing coding answer scores.

### Results

The Decision Tree Regressor outperforms the other models, exhibiting the lowest Mean Squared Error (MSE) and Mean Absolute Error (MAE). This indicates more accurate and consistent predictions. Additionally, its high R-squared (R²) and Explained Variance Score (EVS) values demonstrate its ability to effectively explain the variance in the target variable (answer scores).

### Objectives Achieved

This project successfully:

* **Demonstrates the applicability of NLP and machine learning in analyzing coding answers.**  
* **Evaluates the effectiveness of different language encoding techniques for this task.**
* **Compares the performance of various regression models for predicting answer scores.**
* **Identifies the Decision Tree Regressor as the most suitable model for this dataset.**
* **Provides insights into the factors influencing coding answer quality.** 

### Getting Started

This notebook requires the following Python libraries:

* pandas
* nltk
* gensim
* scikit-learn
* matplotlib

You can install these libraries using pip:
```bash
pip install pandas nltk gensim scikit-learn matplotlib
```

### Future Work

* **Dataset Expansion:**  Increasing the size and diversity of the dataset (more answers, question types, and features) could improve model performance.
* **Advanced NLP Techniques:**  Experimenting with more sophisticated NLP techniques, such as deep learning models (LSTMs, Transformers), to potentially further improve accuracy.
* **Hyperparameter Optimization:**  Utilizing techniques like grid search or random search to fine-tune model hyperparameters for optimal predictions.


