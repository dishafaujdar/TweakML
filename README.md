# 🧠 ML Playground

A modern web application for learning and experimenting with machine learning models and hyperparameters.

## ✨ Features

- **Interactive Learning**: Experiment with 5 different datasets and multiple ML models
- **Real-time Visualization**: See results with interactive charts and metrics
- **Hyperparameter Tuning**: Adjust model parameters with intuitive sliders and controls  
- **Modern UI**: Built with React, TypeScript, TailwindCSS, and shadcn/ui components
- **Fast Backend**: Powered by FastAPI with scikit-learn for ML capabilities

## 🚀 Quick Start

### Prerequisites
- Node.js (v16+)
- Python (v3.8+)
- pip

### Installation & Setup

1. **Navigate to the project directory**
   ```bash
   cd /home/disha/winterArc/TweakML
   ```

2. **Verify the system** (optional but recommended)
   ```bash
   ./verify.sh
   ```

3. **Start the application**
   ```bash
   ./start.sh
   ```

   Or manually start each service:
   - Backend: `cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8001`
   - Frontend: `cd frontend && npx vite`

4. **Open your browser**
   - Go to: **http://localhost:5173**
   - Enter your email to access the playground

### Running the Application

#### Option 1: Quick Start (Recommended)
```bash
cd /home/disha/winterArc/TweakML
./start.sh
```

#### Option 2: Manual Start

1. **Start the Backend** (Terminal 1)
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

2. **Start the Frontend** (Terminal 2)
   ```bash
   cd frontend
   npx vite
   ```

3. **Access the App**
   - Open your browser and go to: http://localhost:5173
   - Enter your email to access the playground

> **Note**: The backend runs on port 8001 and frontend on port 5173. Make sure these ports are available.

## 📊 Available Datasets

- **🌸 Iris**: Classic flower classification (4 features, 3 classes)
- **🍷 Wine Quality**: Wine classification with multiple features
- **🚢 Titanic**: Binary survival prediction with mixed data types
- **🏠 California Housing**: Regression for house price prediction
- **✏️ Handwritten Digits**: Image classification (simplified features)

## 🤖 Supported Models

### Classification
- Decision Tree
- Random Forest  
- Logistic Regression
- Support Vector Machine (SVM)

### Regression
- Decision Tree Regressor
- Random Forest Regressor
- Linear Regression
- Support Vector Regression (SVR)

### Clustering
- K-Means Clustering

## 🎛️ Hyperparameter Controls

Each model includes relevant hyperparameters with intuitive controls:

- **Sliders** for numeric values (max_depth, n_estimators, C, etc.)
- **Dropdowns** for categorical choices (kernel, penalty, init)
- **Toggles** for boolean parameters (fit_intercept)

## 📈 Visualizations

- **Classification**: Confusion matrix heatmaps, feature importance charts
- **Regression**: Feature importance, performance metrics
- **Clustering**: Cluster centers visualization, silhouette analysis

## 🛠️ Tech Stack

### Frontend
- **React 18** with TypeScript
- **TailwindCSS** for styling
- **shadcn/ui** components
- **Plotly.js** for interactive charts
- **Vite** for fast development

### Backend  
- **FastAPI** for API endpoints
- **scikit-learn** for ML models
- **pandas** for data handling
- **NumPy** for numerical operations

## 🔧 API Endpoints

- `GET /datasets` - Get all available datasets with previews
- `GET /models` - Get available models and their hyperparameters
- `POST /train` - Train a model with specified parameters

## 🎯 Usage Flow

1. **Landing Page**: Enter email to access playground
2. **Select Dataset**: Choose from 5 pre-loaded datasets with preview
3. **Choose Model**: Pick appropriate model based on task type
4. **Tune Parameters**: Adjust hyperparameters with interactive controls
5. **Train & View**: See real-time results with metrics and charts

## 🤝 Contributing

This is a V0 implementation focused on core functionality. Future enhancements could include:

- More datasets and models
- Advanced visualization options  
- Model comparison features
- Export capabilities
- User authentication

## 📄 License

MIT License - feel free to use and modify for learning purposes.

---

**Built with ❤️ for ML education and experimentation**
# TweakML
