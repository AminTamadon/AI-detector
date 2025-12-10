# 🧠 AI vs Human Scientific Abstract Detector  
A Streamlit web application that classifies scientific abstracts as **Human-written** or **AI-generated (ChatGPT/DeepSeek)** using a machine-learning model trained on a dataset of 150 scientific abstracts.

## 🚀 Live Application  
Click below to try the app:

👉 **https://YOUR-STREAMLIT-APP-URL-HERE**

---

## 📄 Overview  
This project uses:

- **Sentence embeddings** (MiniLM-L6-v2)
- **Stylometric (writing style) features**
- **Logistic Regression classifier**

The model was trained on:

- 50 Human-written abstracts  
- 50 ChatGPT-generated abstracts  
- 50 DeepSeek-generated abstracts  

The final dataset contained text + 30+ annotation features (clarity, coherence, AI-likelihood, structure, etc.).

The binary model distinguishes:

- **AI** (ChatGPT or DeepSeek)  
- **Human**  

with high accuracy.

---

## 🧰 Features of the App  
✔ Paste any scientific abstract  
✔ The app computes text embeddings  
✔ Adds stylometric linguistic features  
✔ Predicts:  
   - **Human-written** or  
   - **AI-generated**  
✔ Displays prediction probabilities  
✔ Easy deployment through Streamlit Cloud  

---

## 🧪 Machine Learning Model  
The ML pipeline includes:

- **SentenceTransformer** for semantic embeddings  
- **NLTK** for stylometry  
- **Logistic Regression** as classifier  
- **Topic-aware train/test split** (to avoid leakage)  

### Feature Matrix  
- 384-dimensional embedding  
- 7 stylometric features  
➡️ **Total: 391 features**

---

## 📦 Files in This Repository  

