# AI-Powered Plant Disease Pre-Symptom Detection System

This project implements an AI-powered system for detecting plant diseases before symptoms become visible using Convolutional Neural Networks (CNNs). The system analyzes images of plant leaves to identify potential diseases, enabling early intervention and better crop management.

## Project Structure

```
ai-plant-disease-detection
├── src
│   ├── model.py          # Defines the CNN architecture for plant disease detection
│   ├── train.py          # Contains the training logic for the CNN model
│   ├── predict.py        # Handles the prediction process for identifying diseases
│   └── utils.py          # Utility functions for image preprocessing and dataset management
├── data
│   ├── raw               # Directory for storing raw dataset of plant leaf images
│   └── processed         # Directory for storing processed dataset for training
├── models                # Directory for saving trained model files
├── notebooks
│   └── analysis.ipynb    # Jupyter notebook for exploratory data analysis and visualization
├── requirements.txt      # Lists dependencies required for the project
└── README.md             # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd ai-plant-disease-detection
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your raw plant leaf images in the `data/raw` directory. The images should be labeled according to the disease they represent.

2. **Training the Model**: Run the training script to preprocess the images and train the CNN model:

   ```bash
   python src/train.py
   ```

3. **Making Predictions**: Use the prediction script to analyze new images and identify potential diseases:

   ```bash
   python src/predict.py --image_path <path-to-image>
   ```

## Dataset

The dataset used for training and evaluation consists of images of plant leaves, which are categorized based on the type of disease. Ensure that the dataset is well-labeled and organized in the `data/raw` directory.

## Model

The CNN model is designed to learn features from the images and classify them into different disease categories. The architecture can be modified in `src/model.py` to improve performance based on specific requirements.

## Contributing

Contributions to improve the model and enhance the system are welcome. Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for details.