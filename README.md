# Deep-Learning-Predicting-Creditworthiness

Im Rahmen meines Masterseminars wurde ein Projekt zur **Vorhersage der Kreditwürdigkeit** entwickelt. Ziel war es, mithilfe moderner Machine-Learning-Methoden wie **Künstlichen Neuronalen Netzen (Artificial Neural Networks, ANN)** und dem **XGBoost-Algorithmus** präzise Vorhersagen über die Bonität von Kreditnehmern zu treffen. Beide Modelle wurden in **Python** implementiert und auf reale Datensätze angewendet.  

Ein zentraler Aspekt des Projekts war die **Erklärbarkeit der Modelle**, da die Nachvollziehbarkeit von algorithmischen Entscheidungen im Finanzkontext von großer Bedeutung ist. Zur Interpretation der Ergebnisse kamen **SHAP** (*Shapley Additive Explanations*) und **ALE** (*Accumulated Local Effects*) zum Einsatz. Diese Methoden ermöglichen eine detaillierte Analyse der Modellentscheidungen und schaffen Transparenz hinsichtlich der Einflussfaktoren auf die Kreditwürdigkeit.  

Durch den Vergleich der beiden Erklärbarkeitsansätze wurde untersucht, welche Variablen die Kreditwürdigkeit besonders stark beeinflussen und inwiefern sich die zugrunde liegenden Modelle interpretierbar und vertrauenswürdig gestalten lassen.  

---
### Projektdetails
- **Data Source**: [German Credit Data](https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsgermancsv) 
- **Modell**: Neural Network, XGBoost 
- **Verwendete Tools und Bibliotheken**: Implementierung in **Python**  
  - *NumPy*  
  - *Pandas*  
  - *sklearn.preprocessing: StandardScaler* , *sklearn.model_selection: train_test_split*, *sklearn.metrics : accuracy_score, f1_score*
  - *matplotlib.pyplot*  
  - *torch*
  - *shap*
  - *alibi.explainers: ALE*
  - *XGBoost*
---
