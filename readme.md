# 📊 Problematic Internet Use Project

Welcome to the Problematic Internet Use Project! This project is designed to provide robust machine learning model deployment and data processing capabilities. The application is built using FastAPI for serving model predictions and Docker for containerization, making it easy to integrate into various data-driven applications.

## Introduction

A clinical sample of five-thousand 5-22 year-olds who have undergone both clinical and research screenings is provided.

Samples are composed of:
* Physical activity data:
  1. questionnaires.
  2. fitness assessments.
  3. wrist-worn accelerometer data.
* Internet usage behavior data.

The goal of the project is to predict **Severity Impairment Index** (sii), a standard measure of problematic internet use.


## 🚀 Project Structure

The project consists of several components:

- **problematic-internet-use-api**: Contains the FastAPI application for serving model predictions.
- **problematic-internet-use-cache**: A directory used for caching data, such as model files and datasets.
- **problematic-internet-use-job**: Contains scripts related to data preparation, model training, and evaluation.

## 📁 Directory Overview

- `problematic-internet-use-api/`: This directory contains the source code for the FastAPI application.
- `problematic-internet-use-cache/`: This is where cached data and model files are stored.
- `problematic-internet-use-job/`: Includes Python scripts for data preparation, model training, and evaluation.

## 🛠️ Setup Instructions

### Prerequisites

- **Docker**: Ensure you have Docker installed on your machine. You can download it from [Docker's official website](https://www.docker.com/products/docker-desktop).
- **Python 3.8+**: Necessary if you plan to run scripts locally outside of Docker.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd problematic-internet-use
   ```

2. **Build the Docker Image**:
   Navigate to your project directory and build the Docker image using the following command:
   ```bash
   docker-compose build
   ```

### Running the Project

- **Start the API**:
  Run the following command to start the FastAPI application in a Docker container:
  ```bash
  docker-compose up
  ```

  The API will be accessible at `http://localhost:8000`.

### API Usage

- **Endpoint**: `/predict/`
- **Method**: POST
- **Request Body**: JSON object containing the relevant features for model prediction.

Example using `curl`:
```bash
curl -X POST "http://localhost:8000/predict/" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "feature1": 0.32,
  "feature2": 0.92,
  "feature3": -0.34,
  "feature4": -0.62,
  "feature5": 0.97,
  // Add more features as required
}'
```

### 📈 Model Details

- **Algorithm**: XGBoost Classifier (or other specified)
- **Features**: Includes a customizable set of features based on your dataset and model requirements.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## 📜 License

This project is licensed under the MIT License.

## 📨 Contact

For any questions or inquiries, please contact at [badrayour.elamri@protonmail.com](mailto:badrayour.elamri@protonmail.com).

---

Thank you for checking out the Problematic Internet Use Project! 😊
