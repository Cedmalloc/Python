Classifiying Numbers with the MNIST Dataset
- What I learned:
  
  - the accuracy of feedforward NN is not sufficient for classifiying images, CNN is required
  - too many layers as well as too many nodes can lead to overfitting the data
  - it is important to use a seperate validation set to train the hyperparameters (depth,learning rates,...) to avoid overfitting
  - the capacity can be increased/decreased by modifying the amount of layers and nodes as well as the size of the datasets 
    (last part is not always true and depends on the quality of the data, idea is that with more variation more flexibility is required)
  - Training set usually 3 to 4 times larger than validation set
  - just like with finding lanes, reducing the amount of color channels to 1 allows us to process more efficiently especially
    since having color information is not advantageous in this scenario
  - we want to keep the training error low (challenge : underfitting) as well as the gap between training and test error (challenge: overfitting)
