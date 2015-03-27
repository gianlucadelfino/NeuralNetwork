NeuralNetwork
==========
C++ implementation of a Neural Network

Author email: Gianluca.Delfino@gmail.com
<br>
website: www.DoorApps.com

Description
===========
<p>
A fully configurable Neural Network with utility classes to read data from files.
</p>
<p>
An example data set ("TrainingData.txt") representing a XOR operator is provided for testing purposes.
</p>

How to Compile
===========
- Clone repo in the desired directory
  - git clone https://github.com/gianlucadelfino/NeuralNetwork.git
- Run:
  - g++ -Wall -std=c++11 -I./headers main.cpp FileTrainingData.cpp Network.cpp Neuron.cpp XorTrainingData.cpp -o NeuralNetwork.o



License
=======

Released under the MIT license:

* http://www.opensource.org/licenses/MIT
