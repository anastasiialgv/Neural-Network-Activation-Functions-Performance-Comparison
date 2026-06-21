# Neural Network Activation Functions & Performance Comparison

This project focuses on evaluating and comparing the performance of neural network models utilizing different activation functions (**ReLU** and **Tanh**) on structured datasets. It covers the entire machine learning pipeline, including data aggregation, model training, and performance visualization.

## 📁 Project Structure
* `project.py` / `4_2.py` — Core Python scripts handling data preparation, network initialization, and training loops[cite: 2, 3].
* `dane_4_Relu.py` — Script dedicated to configuring and training the network model with the ReLU activation function[cite: 2].
* `dane_7_.py` — Script dedicated to configuring and training the network model with the Tanh activation function[cite: 2].
* `Dane/` — Directory containing individual dataset chunks (`dane1.txt` to `dane16.txt`), alongside specific training and testing data files[cite: 2, 4].
* `dane.csv` / `usdpln.csv` — Supplementary structured comma-separated data files used in analysis[cite: 5].
* `porownanie_modeli.png` — Generated plot visually comparing the performance metrics of the trained models[cite: 4].
* `DataSet_7_tanh.png` — Visualization showcasing the specific behavior and results of the network trained with Tanh[cite: 2].

## 🛠️ Tech Stack
* **Python 3**
* **TensorFlow / Keras** (for building, configuring, and optimizing neural network architectures)
* **Pandas & NumPy** (for high-performance data manipulation and parsing)
* **Matplotlib** (for exporting model comparison plots and matrix visualizations)

## 🚀 Getting Started
1. Install the required data science and deep learning dependencies:
```bash
   pip install tensorflow pandas numpy matplotlib
