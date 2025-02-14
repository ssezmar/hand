# Neural Network Training for Arm Position Classification

This project aims to train a neural network to classify the positions of the wrist, elbow, and shoulder into two categories: fully extended arms (0) and bent arms (1). The dataset consists of 3D coordinates for these points, and the project includes data processing, model training, and exploratory data analysis.

## Project Structure

```
neural-network-training
├── data
│   ├── raw
│   │   └── dataset.csv          # Raw dataset with 3D positions and labels
│   └── processed
│       └── processed_dataset.csv # Processed dataset after normalization
├── models
│   └── model.py                 # Neural network architecture definition
├── notebooks
│   └── data_exploration.ipynb   # Jupyter notebook for data exploration
├── src
│   ├── train.py                  # Main training script
│   └── utils.py                  # Utility functions for data handling
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd neural-network-training
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Data Exploration**: Open the `notebooks/data_exploration.ipynb` to explore the dataset and visualize the 3D positions.

2. **Training the Model**: Run the `src/train.py` script to train the neural network on the processed dataset.

3. **Model Definition**: Modify the `models/model.py` file to adjust the neural network architecture as needed.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.