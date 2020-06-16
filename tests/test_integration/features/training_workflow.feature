Feature: Model training
  I want to be able to train by model when a dataset containing the features and the target is available.

  Scenario: train the model when training data is available
     Given training data is available
      When I launch the training
      Then a model is created in the right folder