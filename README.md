# Pytorch---EVA7
We need to combine an image from MNIST dataset with a randomly generated number. A Neural Network has to be created that will provide 2 outputs:-1) Correct label prediction of the image 2) Sum of the image and the random number.


The model generated 'label 5' as the image from the train_set(MNIST dataset) to be predicted. This will have 10 prediction classes . A list of random numbers was generated to be sent to the network to be compared. The ones which matched showed the result as "corrects". We ran the model twice and the accuracy improved.

<img width="380" alt="Data representation" src="https://user-images.githubusercontent.com/90223404/137543380-d2c14b85-42d2-448c-b05d-8cb2167afb27.png">


Data Generation strategy:-The Random number is getting generated using the 'Range' and 'Random' function in Python. The list of random number is then converted into a trensor to be passed on to the network for comparing with the actual labels. This will help us arrive at the number of correct predictions.



Input Combination:-The image from the MNIST database and the random number have been combined at the output layer.



Result evaluation:-The precitions are compared to the actual labels to arrive at the number of correct predcitions. There are 2 methods to arrive at the number.  1) **prediction.argmax(dim=1).eq(labels)** 2) write a function **def get_num_correct(prediction, labels):    return prediction.argmax(dim=1).eq(labels).sum().item()**.


The Neural Network predicted 2 correct labels. The prediction values were compared with the actual labels to arrive at the result. The more number of times we run the code, the higher the accuracy will be.



Loss function is the method to calculate the prediction error that occurs as part of arriving at correct predictions. We need to minimize this error using an optimization function. We have to deal with the  **Classification** loss in the model since we are dealing with categorical values(predicting 0-9 digits). We have used **Cross Entropy loss** as the loss fucntion as it is the most commonly used and a preferred loss fucntion for classification problems. Another loss function that can be used for binary classification is **Hinge Loss** but for that target values need to be in the set {-1, 1} which is not in our case.


Final Output with 10 epochs

<img width="385" alt="Pytorch Assignment final output" src="https://user-images.githubusercontent.com/90223404/137546118-687d82bc-4aec-49c4-8705-1b0cacac3130.png">
