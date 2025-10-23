## 결과는 나쁘지 않지만 gpu가 놀아서 다시 할 예정

테스트셋 샘플 수: 2023 (cats=1011, dogs=1012)
[CV] Using best fold from cv_summary.csv → ./outputs_catsdogs/best_fold1.pt

Test Classification Report
              precision    recall  f1-score   support

         cat     0.9881    0.9881    0.9881      1011
         dog     0.9881    0.9881    0.9881      1012

    accuracy                         0.9881      2023
   macro avg     0.9881    0.9881    0.9881      2023
weighted avg     0.9881    0.9881    0.9881      2023

Confusion Matrix:
 [[ 999   12]
 [  12 1000]]
 