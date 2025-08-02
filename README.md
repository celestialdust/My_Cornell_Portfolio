# Cornell Machine Learning Program Portfolio

This repository showcases my machine learning coursework and capstone project completed as part of **Cornell University's Machine Learning Certificate Program**. The projects demonstrate proficiency in various ML techniques, from traditional algorithms to deep learning approaches.

## 🎯 Capstone Project: Sentiment Analysis of Book Reviews

**File**: `DefineAndSolveMLProblem.ipynb`

### Project Overview
Built an end-to-end machine learning solution to classify book reviews as positive or negative using natural language processing and neural networks.

### Key Achievements
- **Problem Definition**: Formulated a binary classification problem with clear business value for e-commerce platforms
- **Data Analysis**: Performed comprehensive exploratory data analysis on 1,973 book reviews
- **Feature Engineering**: Implemented TF-IDF vectorization to convert text data into numerical features (18,558 vocabulary size)
- **Model Architecture**: Designed and trained a neural network with:
  - 3 hidden layers (64, 32, 16 neurons)
  - Dropout regularization to prevent overfitting
  - SGD optimizer with binary crossentropy loss
- **Model Performance**: Achieved convergence over 50 epochs with proper validation monitoring
- **Evaluation**: Included comprehensive model evaluation with loss/accuracy visualization

### Technical Implementation
```python
# Model Architecture
nn_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])
```

### Business Impact
This model could help:
- **E-commerce platforms** automatically moderate reviews
- **Publishers** understand reader sentiment at scale  
- **Recommendation systems** improve book suggestions based on review quality

---

## 📚 Additional Course Projects

### Model Selection for K-Nearest Neighbors
**File**: `ModelSelectionForKNN.ipynb`
- Implemented hyperparameter tuning for KNN classification
- Explored the bias-variance tradeoff with different k values
- Demonstrated cross-validation techniques for optimal model selection

### Model Selection for Logistic Regression  
**File**: `ModelSelectionForLogisticRegression.ipynb`
- Applied regularization techniques (L1/L2) to prevent overfitting
- Performed feature selection and coefficient analysis
- Implemented cross-validation for hyperparameter optimization

---

## 🛠 Technical Skills Demonstrated

### **Machine Learning**
- Supervised Learning (Classification)
- Model Selection & Hyperparameter Tuning
- Cross-Validation Techniques
- Bias-Variance Tradeoff Analysis

### **Deep Learning**
- Neural Network Architecture Design
- Regularization Techniques (Dropout)
- Loss Function Selection
- Training Optimization (SGD, Learning Rate Tuning)

### **Natural Language Processing**
- Text Preprocessing
- TF-IDF Vectorization
- Feature Engineering for Text Data

### **Data Science Tools**
- **Python**: NumPy, Pandas, Scikit-learn
- **Deep Learning**: TensorFlow/Keras
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebooks

### **Methodology**
- Complete ML Pipeline Development
- Exploratory Data Analysis
- Model Evaluation & Validation
- Performance Visualization
- Results Interpretation

---

## 📊 Key Results

| Model | Dataset | Accuracy | Key Innovation |
|-------|---------|----------|----------------|
| Neural Network | Book Reviews | Converged Training | TF-IDF + Deep Learning |
| KNN | Various | Cross-Validated | Optimal k Selection |
| Logistic Regression | Various | Regularized | L1/L2 Optimization |

---

## 🏆 Certificate Program Highlights

**Cornell University Machine Learning Certificate**
- Completed comprehensive ML curriculum covering theory and practical applications
- Hands-on experience with real-world datasets
- Focus on industry-relevant techniques and best practices
- Emphasis on model interpretation and business impact

---

## 🚀 Repository Structure

```
My_Cornell_Portfolio/
├── DefineAndSolveMLProblem.ipynb          # 🎯 Capstone: Sentiment Analysis
├── ModelSelectionForKNN.ipynb             # K-Nearest Neighbors Optimization  
├── ModelSelectionForLogisticRegression.ipynb  # Logistic Regression Tuning
├── model_best.pkl                         # Saved Model Artifacts
├── data/                                  # Training Datasets
│   ├── bookReviewsData.csv               # Book Reviews Dataset
│   ├── censusData.csv                    # Census Data
│   ├── airbnbListingsData.csv            # Airbnb Listings
│   └── WHR2018Chapter2OnlineData.csv     # World Happiness Report
└── README.md                             # Project Documentation
```

---

## 💡 Learning Outcomes

Through this program, I developed expertise in:
- **End-to-end ML project lifecycle** from problem definition to deployment-ready models
- **Algorithm selection** based on data characteristics and business requirements  
- **Performance optimization** through systematic hyperparameter tuning
- **Model interpretation** and communication of results to stakeholders
- **Production considerations** including overfitting prevention and validation strategies

---

## 🎓 About Cornell's ML Program

This work was completed as part of Cornell University's rigorous Machine Learning Certificate Program, which emphasizes:
- Theoretical foundations of machine learning algorithms
- Practical implementation using industry-standard tools
- Real-world problem-solving and model deployment
- Statistical rigor in model evaluation and selection

---

**Connect with me**: [LinkedIn](https://linkedin.com/in/xue-jiayu-781b32306) | [GitHub](https://github.com/celestialdust)
