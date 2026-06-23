# User Intent Classification

This repository contains a complete machine-learning project that classifies customer-support messages into 27 intent categories. The data comes from the **Bitext Customer Support LLM Chatbot Training Dataset**.

The application is split into three main components:

1. **Frontend**: A React application (built with Vite and Tailwind CSS).
2. **Backend**: An Express/Node.js server that handles user requests and interacts with a MongoDB database.
3. **ML API**: A FastAPI Python server that serves the machine learning model for intent predictions.

---

## Folder Structure

```text
user-intent-classification/
├── backend/            # Express/Node.js backend API (Routes, Controllers, Models)
├── frontend/           # React frontend application (Components, Views, Styles)
├── ml-api/             # FastAPI service for ML predictions and model files
├── user_intent_classification.py    # Main script for data loading & model training
├── user_intent_classification.ipynb # Jupyter notebook for model exploration
└── README.md
```

## Prerequisites

Make sure you have the following installed:

- **Node.js** (v18+ recommended)
- **Python** (3.8+ recommended)
- **MongoDB** (A local instance or a MongoDB Atlas cluster URL)

---

## Environment Variables Setup

Before running the application, you need to configure the environment variables for the Backend and the ML API.

### 1. Backend Environment Variables

Create a `.env` file in the `backend/` directory and add the following:

```env
# MongoDB Connection String
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>

# Backend Server Port
PORT=5000

# URL of the ML API for predictions
FASTAPI_URL=http://localhost:8000
```

### 2. ML API Environment Variables

Create a `.env` file in the `ml-api/` directory and add the following:

```env
# Port for the FastAPI Server
FASTAPI_PORT=8000
```

---

## How to Start the Application

You will need to open **three separate terminal windows** to run all the components simultaneously.

### Step 1: Start the ML API (Terminal 1)

First, navigate to the `ml-api` folder, install the Python dependencies, and start the FastAPI server.

```bash
cd ml-api

# Optional but recommended: Create and activate a virtual environment
# python -m venv venv
# venv\Scripts\activate      # On Windows
# source venv/bin/activate   # On Mac/Linux

# Install the required packages
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn main:app --reload --port 8000
```

_The ML API will run at `http://localhost:8000`._

### Step 2: Start the Backend Server (Terminal 2)

Next, navigate to the `backend` folder, install the Node.js packages, and start the Express server.

```bash
cd backend

# Install dependencies
npm install

# Start the development server
npm run dev
```

_The Backend API will run at `http://localhost:5000`._

### Step 3: Start the Frontend Application (Terminal 3)

Finally, navigate to the `frontend` folder, install the packages, and start the React application.

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

_The Frontend will be available at `http://localhost:5173` (or whichever port Vite provides)._

---

## Model Training & Exploration (Optional)

If you want to retrain the models or generate new evaluation reports, you can run the root Python scripts:

1. **Install any necessary dependencies** (e.g., pandas, scikit-learn, matplotlib).
2. **Run the training script**:
   ```bash
   python user_intent_classification.py
   ```
   This will train several classic models (Logistic Regression, Random Forest, Decision Tree), evaluate them, and generate evaluation charts (`confusion_matrix.png`, `accuracy_bar_chart.png`), and potentially a LaTeX report.

### Results Summary

- **SVM** achieved the highest accuracy among the three models.
- Over-fitting analysis shows a small gap between train and test accuracy, indicating a well-generalised model.

---

## How to Contribute

If you want to improve the project, feel free to fork the repo, make changes, and submit a pull request. Typical contributions include:

- Adding new models (e.g., SVM, XGBoost).
- Trying deep-learning approaches such as Transformers.
- Extending the dataset to more languages.

## License

This project is released under the MIT License.
