

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
	
#derivative of sigmoid function
def sigmoid_derivative(x):
    return x*(1-x)

#input dataset
training_inputs = open("words.csv", "r").readlines()
print(training_inputs)
# training_inputs=np.array(training_inputs.read().split("\n"))

training_outputs = np.array([[]]).T
#seed random numbers to make calculation deterministic
np.random.seed(1)
synaptic_weights = 2*np.random.random((3,1)) - 1
print('Random starting synaptic weights: ')
print(synaptic_weights)
for iteration in range(10000):
    
    s = "Animation produced outside of Japan with similar style to Japanese animation is commonly referred to as anime-influenced animation. The earliest commercial Japanese animations date to 1917. A characteristic art style emerged in the 1960s with the works of cartoonist Osamu Tezuka and spread in following decades, developing a large domestic audience. Anime is distributed theatrically, through television broadcasts, directly to home media, and over the Internet.  In addition to original works, anime are often adaptations of Japanese comics manga, light novels, or video games. It is classified into numerous genres targeting various broad and niche audiences. Anime is a diverse medium with distinctive production methods that have adapted in response to emergent technologies. It combines graphic art, characterization, cinematography, and other forms of imaginative and individualistic techniques. Compared to Western animation, anime production generally focuses less on movement, and more on the detail of settings and use of camera effects, such as panning, zooming, and angle shots. Diverse art styles are used, and character proportions and features can be quite varied, with a common characteristic feature being large and emotive eyes. The anime industry consists of over 430 production companies, including major studios such as Studio Ghibli, Sunrise, Bones, Ufotable, MAPPA, CoMix Wave Films and Toei Animation.  Since the 1980s, the medium has also seen international success with the rise of foreign dubbed, subtitled programming and its increasing distribution through streaming services.  As of 2016, Japanese animation accounted for 60% of the worlds animated television shows."
    training_inputs = s.split()
    print(training_inputs)
    training_inputs = []
    for i in training_inputs:
     training_inputs.append(float(i))

	
    synaptic_weights = 2*np.random.random((0,1)) - 1

    input_layer = training_inputs
    training_inputs = np.squeeze(np.array(training_inputs))

    #output_layer = output_layer.T
    #input_layer = input_layer.T
    outputs = sigmoid(np.dot(training_inputs.T, synaptic_weights))
   #calculate the error
    error = training_outputs - outputs

    #multiply error by input and gradient of the sigmoid function
    #less confident weights are adjusted more through the nature of the function
    adjustments = error * sigmoid_derivative(outputs)

    #update weights
    synaptic_weights += np.dot(input_layer, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print('Outputs after training: ')
print(outputs)

#test the neural network with a new situation
new_inputs = np.array([])
output = sigmoid(np.dot(new_inputs, synaptic_weights))

print('New situation: ')
print(output)

#input from user

while(1):

	user_input = input("Enter a word: ")
	
	#user_input = np.array([user_input])
	user_input = "test, lol haha"
	user_input = user_input.split()
	print(user_input)

	user_input = []
	
	for i in user_input:
		user_input.append(float(i))
	
	
	synaptic_weights = 2*np.random.random((0,1)) - 1

	output = sigmoid(np.dot(user_input, synaptic_weights))
	print("New input: ")
	print(user_input)
	print("Output after training: ")
	print(output)

