# 📊 Problematic Internet Use Job

Welcome to the Problematic Internet Use Job! This project is designed to automate the preparation, training, and evaluation of machine learning models using Python scripts and Docker for easy deployment and execution.

## 🚀 Project Structure

The project consists of three main components:

1. **Data Preparation**: Download and prepare the dataset for training.
2. **Model Training**: Train a machine learning model based on the specified task.
3. **Model Evaluation**: Evaluate the model's performance using various metrics.

## 📂 Repository Structure

- `main.py`: The main entry point for executing the prepare, train, and evaluate commands.
- `tasks/`: Contains all tasks related to data preparation, training, and evaluation.
  - `tasks/prepare.py`: Prepares the dataset by downloading and processing it.
  - `tasks/train.py`: Trains the Problematic Internet Use model using the prepared dataset.
  - `tasks/evaluate.py`: Evaluates the trained model's performance.
  - `tasks/submit.py`: Submits the csv result file to Kaggle.
- `docker-compose.yml`: Manages the Docker services for each task.

## 🛠️ Setup Instructions

### Prerequisites

- Docker
- Python 3.8+

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd problematic-internet-use-job
   ```

2. **Build the Docker image**:
   ```bash
   docker-compose build
   ```

### Running the Project

Choose a model to train. The model can be set using the environment variable `MODEL_NAME`.


1. **Prepare the Data**:
   ```bash
   docker compose up problematic-internet-use-job-prepare
   ```

2. **Train the Model**:
   ```bash
   docker compose up problematic-internet-use-job-train
   ```

3. **Evaluate the Model**:
   ```bash
   docker compose up problematic-internet-use-job-evaluate
   ```

4. **Evaluate the Model**:
   ```bash
   docker compose up problematic-internet-use-job-submit
   ```

## 📝 Model Details

- **Algorithms**: [XGBoost Classifier, Linear Regressor]
- **Data Handling**: Techniques such as SMOTE can be used to balance the dataset during the preparation phase.
- **Evaluation Metrics**: Common metrics like Accuracy, Precision, Recall, and F1 Score.

## 📈 Output

- **Model**: Saved as `model.ubj` or `model.pkl` in the specified cache directory.
- **Evaluation Plot**: Visualization of model performance (e.g., log loss or other metrics) saved for analysis.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## 📜 License

This project is licensed under the MIT License.

## 📨 Contact

For any questions or inquiries, please contact at [badrayour.elamri@protonmail.com](mailto:badrayour.elamri@protonmail.com).

---

Thank you for checking out the Problematic Internet Use Project! 😊
