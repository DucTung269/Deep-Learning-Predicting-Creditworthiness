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

#### Erklärbarkeit der ANN-Modellausgabe mit Shapley Additive Explanations (SHAP) 

![Shap](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/ExplainShap.png?raw=true)

Der **SHAP-Summary-Plot** zeigt, welche Merkmale den größten Einfluss auf die Modellvorhersage haben. In der linken Spalte sind alle Merkmale aufgelistet, wobei das **wichtigste Merkmal oben** steht. In diesem Fall stellt **Account_Balance** den stärksten Einflussfaktor dar, gefolgt von der **Length_of_current_employment**, **Value_Savings_Stocks** sowie **Payment_Status_of_Previous_Credit**. Das **Occupation** hat hingegen den geringsten Einfluss auf die Vorhersage.  

Die **x-Achse** stellt die **SHAP-Werte** dar, welche den Einfluss eines Merkmals auf die Modellvorhersage quantifizieren. Jeder Punkt im Diagramm repräsentiert eine Beobachtung im Datensatz und zeigt, ob das entsprechende Merkmal einen **positiven oder negativen Einfluss** auf die Vorhersage hatte.  

Die **Farben der Punkte** geben an, ob das jeweilige Merkmal einen **hohen (rot)** oder **niedrigen (blau)** Wert aufweist.  
- Hohe Kontostände (rote Punkte) erhöhen die Wahrscheinlichkeit einer positiven Kreditentscheidung und wirken sich somit **förderlich auf die Kreditwürdigkeit** aus.  
- Niedrige Kontostände (blaue Punkte) verringern diese Wahrscheinlichkeit und wirken **negativ auf die Kreditwürdigkeit**.  

Ein ähnliches Muster lässt sich bei den Merkmalen **Length_of_current_employment**, **Value_Savings_Stock**, **Payment_Status_of_Previous_Credit** und **Telephone** beobachten. Umgekehrt zeigt sich, dass eine **geringe Anzahl bestehender Kredite bei der gleichen Bank** einen positiven Einfluss hat, während eine hohe Kreditanzahl die Wahrscheinlichkeit einer positiven Bewertung verringert.  

Die **Dichte der Punkte entlang der x-Achse** zeigt, wie häufig bestimmte SHAP-Werte auftreten – eine hohe Dichte bedeutet, dass viele Beobachtungen einen ähnlichen Einfluss des Merkmals auf die Vorhersage aufweisen. Die **vertikale Streuung** eines Merkmals deutet auf eine **starke Wechselwirkung mit anderen Variablen** hin.  

#### Visuelle Erklärbarkeit der ANN-Modellausgabe mit Accumulated Local Effects (ALE)

![ALE](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/ExplainALE.png?raw=true)

Die Analyse der **ALE-Plots (Accumulated Local Effects)** ermöglicht ein tieferes Verständnis darüber, welche Merkmale die Vorhersagen des neuronalen Netzes beeinflussen und wie sich diese Effekte über verschiedene Merkmalsausprägungen hinweg verändern. Dadurch wird der **Entscheidungsprozess des Modells** transparenter und nachvollziehbarer.  

In der Abbildung ist zu erkennen, dass **alle Merkmale** Einfluss auf die Vorhersageergebnisse des **ANN-Modells** haben, was sich in den dargestellten **linearen oder nichtlinearen (gekrümmten) Mustern** der ALE-Werte widerspiegelt.  

Zur genaueren Betrachtung werden drei zentrale Merkmale hervorgehoben: **Alter (Age_years)**, **Kontostand (Account_Balance)** und **Kreditbetrag (Credit_Amount)**.  

- Beim Merkmal **Age_years** zeigen die ALE-Werte, dass **jüngere Personen** ein **höheres Risiko eines Kreditausfalls** aufweisen (negative ALE-Werte), während **mittelalte und ältere Personen** ein geringeres Risiko zeigen. Dieses Muster lässt sich durch eine häufig höhere **finanzielle Stabilität im höheren Alter** erklären.  

- Für den **Account_Balance** (Wertebereich 1–4) zeigen die ALE-Werte, dass Personen mit einem **Kontostand größer als 2** ein **deutlich reduziertes Ausfallrisiko** haben. Ein höherer Kontostand steht somit in starkem Zusammenhang mit **finanzieller Zuverlässigkeit**.  

- Beim **Credit_Amount** zeigt sich eine intuitive Beziehung: Mit steigendem Kreditbetrag erhöhen sich auch die ALE-Werte, was auf ein **höheres Ausfallrisiko** hinweist. Größere Kreditsummen bedeuten eine **höhere finanzielle Belastung**, wodurch das Risiko eines Zahlungsausfalls zunimmt.  


#### XGBoost

![XGB](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/XGB.jpg?raw=true)

- Das **XGBoost-Modell** wurde für den **Klassifikationsfall** unter Verwendung der gleichen Trainingsdaten wie das **Artificial-Neural-Network-Modell (ANN)** eingesetzt.  

- Die **Leistungsbewertung** des XGBoost-Modells zeigt, dass es ähnlich gute Ergebnisse wie das ANN erzielt hat. Auf dem **Trainingsdatensatz** erreichte das Modell eine **Genauigkeit von 78,3 %** und einen **F1-Score von 79,62 %**. Auf dem **Testdatensatz** wurden eine **Genauigkeit von 73,5 %** und ein **F1-Score von 81 %** erzielt.  

- Diese Ergebnisse deuten darauf hin, dass beide Modelle eine **gute Generalisierungsfähigkeit** aufweisen, wobei sich lediglich **geringfügige Unterschiede in der Vorhersageleistung** zeigen.  

#### Erklärbarkeit der XGBoost-Modellausgabe mit Shapley Additive Explanations (SHAP) 

![SHAP XGB](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/SHAP%20for%20XGB.png?raw=true)

#### Visuelle Erklärbarkeit der XGBoost-Modellausgabe mit Accumulated Local Effects (ALE)

![ALE XGB](https://github.com/DucTung269/Deep-Learning-Predicting-Creditworthiness/blob/main/images/ALE%20for%20XGB.png?raw=true)

- In der Abbildung zeigen die Merkmale **No_of_dependents**, **Type_of_apartment**, **Foreign_Workers** und **Occupation** nur einen **geringen oder keinen erkennbaren Einfluss** auf die Modellvorhersage. Ihre **ALE-Werte bleiben über alle Merkmalsausprägungen hinweg nahezu konstant**, was darauf hindeutet, dass diese Variablen **keinen signifikanten Beitrag** zu den Entscheidungen des Modells leisten.  

- Die **ALE-Plots des XGBoost-Modells** zeigen, dass die drei Merkmale **Alter (Age_years)**, **Kontostand (Account_Balance)** und **Kreditbetrag (Credit_Amount)** ähnliche Ergebnisse liefern wie beim **ANN-Modell**.  

- Allerdings lässt sich beim Merkmal **Age_years** ein leichter Unterschied erkennen: **Personen mittleren Alters** weisen laut den ALE-Werten eine **etwas höhere Wahrscheinlichkeit auf, einen Kredit aufzunehmen**, als ältere Personen.  

- Beim **Account_Balance** bleibt der Trend konsistent: Individuen mit einem **Kontostand größer als 2** zeigen ein **deutlich geringeres Ausfallrisiko**, was erneut die enge **Korrelation zwischen höherem Kontostand und finanzieller Stabilität** bestätigt.  

- Ebenso zeigt das Merkmal **Credit_Amount**, dass **höhere Kreditsummen mit einem erhöhten Risiko eines Zahlungsausfalls** einhergehen — ein Ergebnis, das den allgemeinen wirtschaftlichen Erwartungen entspricht.  

---



