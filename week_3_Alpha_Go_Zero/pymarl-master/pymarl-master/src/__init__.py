'''
Runner <----> Controller <----> Modules (networks)
   |                              ^
   |                              |
   |-----> Learner <---------------

The Runner is responsible for environment interaction and data collection, and it calls the Controller to select actions.

The Controller is responsible for multi-agent action decision-making. It calls the models in Modules to perform forward inference and decide actions.

The Learner is responsible for training the agent model based on the collected data. It receives data batches from the Runner,
computes the loss function, performs backpropagation and gradient updates, and optimizes the model parameters in the Modules.

Modules define the specific neural network architecture.
The Learner updates the model parameters in Modules using the data, and the updated parameters affect the Controller’s action selection.

'''