# Deep-Learning-Predicting-Creditworthiness

Im Rahmen meines Masterseminars wurde ein Projekt zur **Vorhersage der Kreditwürdigkeit** entwickelt. Ziel war es, mithilfe moderner Machine-Learning-Methoden wie **Künstlichen Neuronalen Netzen (Artificial Neural Networks, ANN)** und dem **XGBoost-Algorithmus** präzise Vorhersagen über die Bonität von Kreditnehmern zu treffen. Beide Modelle wurden in **Python** implementiert und auf reale Datensätze angewendet.  

Ein zentraler Aspekt des Projekts war die **Erklärbarkeit der Modelle**, da die Nachvollziehbarkeit von algorithmischen Entscheidungen im Finanzkontext von großer Bedeutung ist. Zur Interpretation der Ergebnisse kamen **SHAP** (*Shapley Additive Explanations*) und **ALE** (*Accumulated Local Effects*) zum Einsatz. Diese Methoden ermöglichen eine detaillierte Analyse der Modellentscheidungen und schaffen Transparenz hinsichtlich der Einflussfaktoren auf die Kreditwürdigkeit.  

Durch den Vergleich der beiden Erklärbarkeitsansätze wurde untersucht, welche Variablen die Kreditwürdigkeit besonders stark beeinflussen und inwiefern sich die zugrunde liegenden Modelle interpretierbar und vertrauenswürdig gestalten lassen.  

---
### Projektdetails  

- **Datenquelle**: [German Credit Data](https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsgermancsv)  
- **Modelle**: Neuronales Netz (Artificial Neural Network) und XGBoost  
- **Verwendete Tools und Bibliotheken**: Implementierung in **Python**  
  - *NumPy*  
  - *Pandas*  
  - *sklearn.preprocessing*: *StandardScaler*  
  - *sklearn.model_selection*: *train_test_split*  
  - *sklearn.metrics*: *accuracy_score*, *f1_score*  
  - *matplotlib.pyplot*  
  - *torch*  
  - *shap*  
  - *alibi.explainers*: *ALE*  
  - *XGBoost*  

---
### Ergebnisseübetsicht
- **Code**: Der Python-Code zur Implementierung der Datenvorverarbeitung, des **Artificial-Neural-Network-Modells** sowie der Ergebnisinterpretation befindet sich im Jupyter Notebook *Predicting Creditworthiness.ipynb* in [meinem Repository](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness).  
- **Dataset and Data Preprocessing**: Der Datensatz umfasst **1000 Beobachtungen** mit **21 Variablen**. Die meisten dieser Eingangsvariablen sind kategorial und repräsentieren unterschiedliche Zustände oder Ausprägungen einer bestimmten Eigenschaft. Der Datensatz enthält grundlegende Informationen über Kreditnehmer, wie beispielsweise **Kreditbetrag**, **Familienstand**, **Alter**, **Beschäftigungsstatus**, **Kreditzweck** sowie die Information, ob der Kreditnehmer **im Ausland beschäftigt** ist.  

  Die Zielvariable **Creditability** dient der Klassifikation der Kreditwürdigkeit einer Person. Ein Wert von **1** steht für *kreditwürdig* (kein Zahlungsausfall), während ein Wert von **0** *nicht kreditwürdig* bedeutet (Zahlungsausfall).  
![](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/imbalance%20dataset.jpg?raw=true)
---
