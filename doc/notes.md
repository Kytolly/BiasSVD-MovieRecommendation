# notes when traning

## explanation
`underlying_features_K` : the number of the movie and user underlying features; 
`steps` :  the total turns of iteration;
`lambda_r`:  the weight of regulation to avoid overfitting; 
`learning_rate`: used in stochastic gradient descent(SGD) optimizer;

## 2024-11-6(16:00)
the config.yaml follows:
```yaml
underlying_features_K : 3 
steps: 1000              
lambda_r: 0.0001             
learning_rate: 0.001    
```

the log of validating and testing:
```log
start training...

the turn 1 start!
the turn 1 finished, the validating result is: 2.2667240916981313
the turn 2 start!
the turn 2 finished, the validating result is: 2.73755048262131
the turn 3 start!
the turn 3 finished, the validating result is: 2.5779425204096302
the turn 4 start!
the turn 4 finished, the validating result is: 2.1075715809976154
the turn 5 start!
the turn 5 finished, the validating result is: 2.221574487210751
the turn 6 start!
the turn 6 finished, the validating result is: 1.6077776285630778
the turn 7 start!
the turn 7 finished, the validating result is: 2.098580216285282
the turn 8 start!
the turn 8 finished, the validating result is: 1.9177965818470697
the turn 9 start!
the turn 9 finished, the validating result is: 1.7439211217601855
start tesing...

the testing result is: 1.954402138442144
```