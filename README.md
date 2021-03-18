## SocialSigNet

### Files:
1) working.py: code to create social signatures
2) createFootprints.py: code to create images of each row in us_migration.csv based on trained weights from working.py
3) resnsetWorking.py: uses slightly modified resnset18 archtiecture to train model based on social signature images created in createFootprints.py

<br>
<br>

### Current Results


|      Model     |      MAE	    |	 MAPE   | 	    R2       |	 PPT
|----------------|--------------|-----------|----------------|--------------
| socialSigNet	 |	185.35498	| 13.829773	| -0.108545035	 | 11574.14167
| Neural Network |	211.4892987	| 2.68E+16	|  0.03873911901 | 126271.7745
| Decision Tree	 |	199.0784946	| 3.34E+17	| -0.8513310438	 | 126331.7323
| Random Forest	 |	195.0663231	| 1.72E+18	|  0.1093394465	 | 126268.1216
| KNN		     |  258.3322581	| 2.47E+18	| -0.8390059399	 | 126219.7086

<br>
<br>

PPT = abs(sum(y_true) - sum(y_pred)) / len(y_true)