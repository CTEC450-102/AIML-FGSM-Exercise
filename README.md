# AI Model Attack and Defense (MNIST - PyTorch)

## Overview

This project demonstrates the full lifecycle of a machine learning system under adversarial conditions. It includes:

* Training a Convolutional Neural Network (CNN) on the MNIST handwritten digits dataset
* Performing an adversarial attack using the Fast Gradient Sign Method (FGSM)
* Evaluating the degradation in model performance under attack
* Implementing (optional) adversarial defense techniques to improve robustness

The goal is to highlight how vulnerable AI models can be to adversarial inputs and how defensive strategies can mitigate those risks.

===================================================================================================

## Project Structure

```
.
├── data/                  # Dataset storage (auto-downloaded)
├── main.py               # Main Python script (your code)
└── README.md             # Project documentation
```

===================================================================================================

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

===================================================================================================

## Dataset

This project uses the **MNIST dataset**, which consists of 28x28 grayscale images of handwritten digits (0–9).

* Training samples: 60,000
* Test samples: 10,000

The dataset is automatically downloaded when the script is executed.

===================================================================================================

## Model Architecture

A Convolutional Neural Network (CNN) is implemented with:

* 2 Convolutional layers
* Dropout layer for regularization
* 2 Fully connected layers
* ReLU activation functions
* Log Softmax output layer

===================================================================================================

## Phases of the Project

### 1. Training Phase

The model is trained on the MNIST dataset using:

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Epochs: 10

During training:

* Forward propagation computes predictions
* Loss is calculated
* Backpropagation updates model weights

===================================================================================================

### 2. Attack Phase (FGSM)

The Fast Gradient Sign Method (FGSM) is used to generate adversarial examples.

#### Key Idea:

Small perturbations are added to input images to trick the model into making incorrect predictions.

#### Formula:

```
perturbed_image = image + epsilon * sign(gradient)
```

#### Epsilon Values Used:

```
[0, 0.05, 0.1, 0.15, 0.2, 0.25]
```

As epsilon increases:

* Noise increases
* Model accuracy decreases

===================================================================================================

### 3. Defense Phase (Adversarial Training)

A defense strategy is included (commented out in code):

* Adversarial examples are generated during training
* Model is retrained using both clean and adversarial data
* This improves robustness against attacks

===================================================================================================

## How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision matplotlib numpy
```

---

### 2. Run the Program

```bash
python main.py
```

===================================================================================================

## Output

### Training Output

* Loss values printed during training
* Accuracy on training dataset

### Attack Output

* Accuracy decreases as epsilon increases
* Printed results for each epsilon level

### Visualizations

1. **Accuracy vs Epsilon Graph**

   * Shows how model performance degrades under attack

2. **Adversarial Examples Grid**

   * Displays original vs perturbed predictions
   * Shows how small changes fool the model

===================================================================================================

## Key Functions

### `train(epoch)`

Trains the model for one epoch.

### `test()`

Evaluates model performance on training data.

### `fgsm_attack(image, epsilon, data_grad)`

Generates adversarial examples using FGSM.

### `trainAttack(model, device, loader, epsilon)`

Runs the attack and computes accuracy under adversarial conditions.

===================================================================================================

## Results Summary

* Model performs well on clean MNIST data
* Accuracy drops significantly under adversarial attack
* Higher epsilon values lead to greater performance degradation
* Adversarial training (if enabled) improves robustness

===================================================================================================

## References

* Goodfellow, I., Shlens, J., & Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples*
* NeuralNine (2023). *PyTorch Project: Handwritten Digit Recognition*
* MLWorks (2026). *Adversarial Attacks in Machine Learning*

===================================================================================================

## Author

**Raynard Forte II**
CTEC 450-102
Final Project – AI Model Attack and Defense
April 30, 2026

===================================================================================================

## Notes

* GPU support is enabled if CUDA is available
* The defense phase is currently commented out in the code
* You can enable it to observe improved robustness

===================================================================================================

## Future Improvements

* Implement additional attack methods (PGD, DeepFool)
* Evaluate on more complex datasets (CIFAR-10)
* Improve visualization of adversarial noise
* Add test dataset evaluation instead of training-only testing

===================================================================================================

##AI Use Policy:
 
    • Use Copilot for: 
        ◦ boilerplate, CLI parsing, JSON formatting, unit test scaffolds 
    • Do not ask Copilot for: 
        ◦ Attack public or private AI Models of others 
        ◦ bypassing OS permissions 
    • Always: 
        ◦ Use public source data or ask for permission
     
        
