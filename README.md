# Hall Sensors
This project attempts to charaterize the sensitivity of novel hall sensors

## Installation
In order to install this project simply download all the files to your machine. Then, open a terminal, navigate to the folder containing this project, and run 
```
pip install -r requirements.txt
```
in order to download dependencies. Now you are ready to use this project. 

## Features
This project finds the parameters that will fit the data to my model of the system. There are two functions that can be used to do this fitting. The first is 
```
analyze_device('Name of Device')
```
This function will fetch the data for the device, perform the fitting, and produce graphs for this device. The other function is 
```
analyze_all()
```
This function will fetch the data for all devices, perform the fitting, and produce graphs with all devices. 

## Acknowledgments
I would like to acknowledge the Molecular Beam Epitaxy group at the University of Oklahoma and their partner Amethyst Technologies for overseeing this project. 

## Future Work
I will not be performing future work on this project. However, the main open question in the project is why the Epilayer has a smaller error than the Quantum Well devices when our intuition of the physics tells us the opposite should be true. 
