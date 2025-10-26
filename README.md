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

![ ](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/imbalance%20dataset.jpg?raw=true)

- Wie in der Abbildung dargestellt, ist die Anzahl der **not default** (kein Zahlungsausfall) deutlich höher als die der **default** (Zahlungsausfall) (700 gegenüber 300). Um dieses **Klassenungleichgewicht** zu adressieren, wurde ein entsprechender Schritt der **Datenvorverarbeitung** implementiert.  

- Zur Korrektur des Ungleichgewichts wurde die Methode der **Class Weights** angewendet. Dabei erhalten Beobachtungen der Mehrheitsklasse ein geringeres Gewicht, während Beobachtungen der Minderheitsklasse stärker gewichtet werden. Auf diese Weise wird das Modell dazu veranlasst, der Minderheitsklasse während der Berechnung der Verlustfunktion mehr Aufmerksamkeit zu schenken, was zu einer ausgewogeneren Modellleistung führt.  

![classweight](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/classweight.jpg?raw=true)

- Darüber hinaus kam während der Datenvorverarbeitung der **StandardScaler()** zum Einsatz. Dieses Verfahren standardisiert die Eingangsvariablen, indem es sie so transformiert, dass sie einen Mittelwert von 0 und eine Standardabweichung von 1 aufweisen. Durch diese **Normalisierung** befinden sich alle Merkmale auf einer vergleichbaren Skala, was **die Trainingsstabilität verbessert** und **eine Verzerrung zugunsten großskaliger Variablen verhindert**. Somit trägt die Standardisierung wesentlich dazu bei, dass alle Merkmale einen gleichberechtigten Einfluss auf die Modellvorhersage haben.  

#### Artificial Neural Network
![ANN Code](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/ANN%20Code.jpg?raw=true)

- Ein **Artificial Neural Network (ANN)** wurde wurde implementiert, das aus einer **Eingabeschicht (Input Layer)**, mehreren **verdeckten Schichten (Hidden Layers)** und einer **Ausgabeschicht (Output Layer)** besteht.  

- Die Eingabeschicht erhält die Eingangsmerkmale des Datensatzes und leitet sie an die verdeckten Schichten weiter. Jede verdeckte Schicht besteht aus einer bestimmten Anzahl von **Neuronen**, die über **lineare Transformationen** (Gewichtungen und Bias) sowie die **ReLU-Aktivierungsfunktion** verarbeitet werden. 

![Hyperparameters](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/Hyperparameter.jpg?raw=true)

- Die **Hyperparameter** des neuronalen Netzes konnten flexibel angepasst und wurden anschließend festgelegt, um das Modell zu initialisieren. Dabei wurden die **Eingabegröße**, die **Anzahl der Neuronen pro Schicht**, die **Anzahl der verdeckten Schichten** sowie die **Ausgabegröße** definiert. Das Modell wurde mit einer **Lernrate von 0.0001** trainiert.  

- Als **Verlustfunktion** wurde `BCEWithLogitsLoss` verwendet, wobei ein Gewichtungsfaktor (*pos_weight*) integriert wurde, um das **Klassenungleichgewicht** im Datensatz auszugleichen. Zur Optimierung der Modellparameter kam der **Adam-Optimierer** zum Einsatz.  

- Für den Trainingsprozess wurden **100 Epochen** und eine **Batchgröße von 128** verwendet.  



![outcome](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/output.png?raw=true)

- Nach **100 Epochen** konvergiert der **Loss-Wert** auf einen stabilen und niedrigen Wert von etwa **0.20**, was darauf hinweist, dass das Modell die zugrunde liegenden Muster der Daten effektiv erlernt hat. Der **gleichmäßig abfallende Verlauf** des Verlusts über die Epochen hinweg zeigt einen **gut funktionierenden Trainingsprozess**.  

- Es sind **keine Anzeichen von Overfitting** erkennbar, das sich typischerweise durch einen Anstieg des Verlusts auf den Validierungsdaten nach einer bestimmten Anzahl von Epochen äußern würde. Ebenso gibt es **keine Hinweise auf Underfitting**, welches sich durch eine frühe Stagnation des Verlusts auf einem höheren Niveau bemerkbar machen würde.  

![F1](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/F1.Score.jpg?raw=true)

- Für den **Trainingsdatensatz** erreichte das Modell eine **Genauigkeit von 77,5 %** und einen **F1-Score von 82,92 %**. Dies zeigt, dass das Modell während der Trainingsphase sowohl in Bezug auf **Präzision** als auch **Sensitivität (Recall)** solide Leistungen erzielte.  

- Beim **Testdatensatz** sank die Genauigkeit leicht auf **75,5 %**, während der **F1-Score mit 80,93 %** weiterhin ein starkes Ergebnis aufweist. Dies deutet darauf hin, dass das Modell seine **Vorhersagefähigkeit auch bei unbekannten Daten beibehalten** konnte, wenngleich mit einem geringfügigen Rückgang der Gesamtgenauigkeit.  

#### XGBoost

![XGB](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/XGB.jpg?raw=true)

- Das **XGBoost-Modell** wurde für den **Klassifikationsfall** unter Verwendung der gleichen Trainingsdaten wie das **Artificial-Neural-Network-Modell (ANN)** eingesetzt.  

- Die **Leistungsbewertung** des XGBoost-Modells zeigt, dass es ähnlich gute Ergebnisse wie das neuronale Netz erzielt hat. Auf dem **Trainingsdatensatz** erreichte das Modell eine **Genauigkeit von 78,3 %** und einen **F1-Score von 79,62 %**. Auf dem **Testdatensatz** wurden eine **Genauigkeit von 73,5 %** und ein **F1-Score von 81 %** erzielt.  

- Diese Ergebnisse deuten darauf hin, dass beide Modelle eine **gute Generalisierungsfähigkeit** aufweisen, wobei sich lediglich **geringfügige Unterschiede in der Vorhersageleistung** zeigen.  


![Shap](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/ExplainShap.png?raw=true)

![ALE](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/ExplainALE.png?raw=true)


---



