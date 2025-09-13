from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, silhouette_score
)
from sklearn.model_selection import learning_curve
from sklearn.datasets import (
    load_iris, load_wine, load_digits,
    fetch_california_housing
)
import json

app = FastAPI(title="ML Playground API", description="Backend for ML Playground")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when allowing all origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class TrainingRequest(BaseModel):
    dataset_name: str
    model_type: str
    hyperparameters: Dict[str, Any]

class DatasetResponse(BaseModel):
    name: str
    shape: List[int]
    features: List[str]
    target: str
    task_type: str
    preview: List[Dict[str, Any]]

# Global datasets storage
DATASETS = {}

def load_datasets():
    """Load all predefined datasets"""
    global DATASETS
    
    # Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    DATASETS['iris'] = {
        'data': iris_df,
        'target_column': 'target',
        'task_type': 'classification',
        'description': 'Iris flower classification'
    }
    
    # Wine dataset
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    DATASETS['wine'] = {
        'data': wine_df,
        'target_column': 'target',
        'task_type': 'classification',
        'description': 'Wine quality classification'
    }
    
    # Digits dataset (simplified)
    digits = load_digits()
    # Use only first 8 features for simplicity
    digits_df = pd.DataFrame(digits.data[:, :8], 
                           columns=[f'pixel_{i}' for i in range(8)])
    digits_df['target'] = digits.target
    DATASETS['digits'] = {
        'data': digits_df,
        'target_column': 'target',
        'task_type': 'classification',
        'description': 'Handwritten digits classification'
    }
    
    # California Housing
    housing = fetch_california_housing()
    housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
    housing_df['target'] = housing.target
    DATASETS['california_housing'] = {
        'data': housing_df,
        'target_column': 'target',
        'task_type': 'regression',
        'description': 'California housing prices prediction'
    }
    
    # Create a synthetic Titanic-like dataset
    np.random.seed(42)
    n_samples = 891
    titanic_data = {
        'Age': np.random.normal(29, 14, n_samples),
        'Fare': np.random.exponential(32, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Sex': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),  # 0=female, 1=male
        'SibSp': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.01]),
        'Parch': np.random.choice([0, 1, 2, 3], n_samples, p=[0.76, 0.13, 0.08, 0.03])
    }
    
    # Create target based on some logic
    titanic_df = pd.DataFrame(titanic_data)
    titanic_df['Age'] = np.clip(titanic_df['Age'], 0, 80)
    
    # Simple survival logic
    survival_prob = (
        0.7 * (titanic_df['Sex'] == 0) +  # Females more likely
        0.3 * (titanic_df['Pclass'] == 1) +  # First class more likely
        0.2 * (titanic_df['Age'] < 16) +  # Children more likely
        -0.1 * (titanic_df['Age'] > 60)  # Elderly less likely
    )
    titanic_df['Survived'] = np.random.binomial(1, np.clip(survival_prob, 0.1, 0.9))
    
    DATASETS['titanic'] = {
        'data': titanic_df,
        'target_column': 'Survived',
        'task_type': 'classification',
        'description': 'Titanic survival prediction'
    }

# Load datasets on startup
load_datasets()

@app.get("/")
async def root():
    return {"message": "ML Playground API is running!"}

@app.get("/datasets", response_model=List[DatasetResponse])
async def get_datasets():
    """Get all available datasets with preview"""
    datasets = []
    for name, dataset_info in DATASETS.items():
        df = dataset_info['data']
        preview = df.head(10).to_dict('records')
        
        datasets.append(DatasetResponse(
            name=name,
            shape=list(df.shape),
            features=list(df.columns[:-1]),  # Exclude target column
            target=dataset_info['target_column'],
            task_type=dataset_info['task_type'],
            preview=preview
        ))
    
    return datasets

@app.get("/models")
async def get_models():
    """Get available models and their hyperparameters"""
    return {
        "classification": {
            "decision_tree": {
                "name": "Decision Tree",
                "hyperparameters": {
                    "max_depth": {"type": "int", "min": 1, "max": 20, "default": 5},
                    "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
                    "min_samples_leaf": {"type": "int", "min": 1, "max": 20, "default": 1}
                }
            },
            "random_forest": {
                "name": "Random Forest",
                "hyperparameters": {
                    "n_estimators": {"type": "int", "min": 10, "max": 200, "default": 100},
                    "max_depth": {"type": "int", "min": 1, "max": 20, "default": 5},
                    "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2}
                }
            },
            "logistic_regression": {
                "name": "Logistic Regression",
                "hyperparameters": {
                    "C": {"type": "float", "min": 0.001, "max": 100, "default": 1.0},
                    "penalty": {"type": "select", "options": ["l1", "l2"], "default": "l2"}
                }
            },
            "svm": {
                "name": "Support Vector Machine",
                "hyperparameters": {
                    "C": {"type": "float", "min": 0.001, "max": 100, "default": 1.0},
                    "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
                    "gamma": {"type": "float", "min": 0.001, "max": 10, "default": 1.0}
                }
            }
        },
        "regression": {
            "decision_tree": {
                "name": "Decision Tree",
                "hyperparameters": {
                    "max_depth": {"type": "int", "min": 1, "max": 20, "default": 5},
                    "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
                    "min_samples_leaf": {"type": "int", "min": 1, "max": 20, "default": 1}
                }
            },
            "random_forest": {
                "name": "Random Forest",
                "hyperparameters": {
                    "n_estimators": {"type": "int", "min": 10, "max": 200, "default": 100},
                    "max_depth": {"type": "int", "min": 1, "max": 20, "default": 5},
                    "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2}
                }
            },
            "linear_regression": {
                "name": "Linear Regression",
                "hyperparameters": {
                    "fit_intercept": {"type": "boolean", "default": True}
                }
            },
            "svm": {
                "name": "Support Vector Machine",
                "hyperparameters": {
                    "C": {"type": "float", "min": 0.001, "max": 100, "default": 1.0},
                    "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
                    "gamma": {"type": "float", "min": 0.001, "max": 10, "default": 1.0}
                }
            }
        },
        "clustering": {
            "kmeans": {
                "name": "K-Means",
                "hyperparameters": {
                    "n_clusters": {"type": "int", "min": 2, "max": 10, "default": 3},
                    "init": {"type": "select", "options": ["k-means++", "random"], "default": "k-means++"},
                    "max_iter": {"type": "int", "min": 100, "max": 1000, "default": 300}
                }
            }
        }
    }

def get_model(model_type: str, task_type: str, hyperparams: Dict[str, Any]):
    """Create model instance based on type and hyperparameters"""
    if task_type == "classification":
        if model_type == "decision_tree":
            return DecisionTreeClassifier(**hyperparams)
        elif model_type == "random_forest":
            return RandomForestClassifier(**hyperparams)
        elif model_type == "logistic_regression":
            return LogisticRegression(**hyperparams, max_iter=1000)
        elif model_type == "svm":
            return SVC(**hyperparams)
    elif task_type == "regression":
        if model_type == "decision_tree":
            return DecisionTreeRegressor(**hyperparams)
        elif model_type == "random_forest":
            return RandomForestRegressor(**hyperparams)
        elif model_type == "linear_regression":
            return LinearRegression(**hyperparams)
        elif model_type == "svm":
            return SVR(**hyperparams)
    elif task_type == "clustering":
        if model_type == "kmeans":
            return KMeans(**hyperparams)
    
    raise ValueError(f"Unsupported model type: {model_type}")

@app.post("/train")
async def train_model(request: TrainingRequest):
    """Train a model and return results"""
    try:
        # Get dataset
        if request.dataset_name not in DATASETS:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_info = DATASETS[request.dataset_name]
        df = dataset_info['data'].copy()
        task_type = dataset_info['task_type']
        target_col = dataset_info['target_column']
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Get model
        model = get_model(request.model_type, task_type, request.hyperparameters)
        
        results = {}
        
        if task_type in ["classification", "regression"]:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task_type == "classification":
                # Classification metrics
                results = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                }
                
                # Generate learning curve
                try:
                    train_sizes, train_scores, val_scores = learning_curve(
                        model, X_train, y_train, cv=3, n_jobs=-1, 
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        scoring='accuracy'
                    )
                    results["training_curve"] = {
                        "train_sizes": train_sizes.tolist(),
                        "train_scores": np.mean(train_scores, axis=1).tolist(),
                        "validation_scores": np.mean(val_scores, axis=1).tolist()
                    }
                except:
                    # Fallback if learning_curve fails
                    pass
                
                # Generate decision boundary for 2D visualization (first 2 features)
                if X.shape[1] >= 2:
                    try:
                        # Use first two features for visualization
                        X_2d = X_train.iloc[:, :2]
                        model_2d = get_model(request.model_type, task_type, request.hyperparameters)
                        model_2d.fit(X_2d, y_train)
                        
                        # Create mesh grid
                        x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
                        y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                           np.linspace(y_min, y_max, 50))
                        
                        # Predict on mesh
                        mesh_points = np.c_[xx.ravel(), yy.ravel()]
                        Z = model_2d.predict(mesh_points)
                        Z = Z.reshape(xx.shape)
                        
                        results["decision_boundary"] = {
                            "x_range": xx[0].tolist(),
                            "y_range": yy[:, 0].tolist(),
                            "predictions": Z.tolist(),
                            "feature_names": [X.columns[0], X.columns[1]]
                        }
                    except:
                        # Fallback if decision boundary fails
                        pass
                    
            elif task_type == "regression":
                # Regression metrics
                results = {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "r2_score": float(r2_score(y_test, y_pred))
                }
                
                # Generate learning curve for regression
                try:
                    train_sizes, train_scores, val_scores = learning_curve(
                        model, X_train, y_train, cv=3, n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        scoring='r2'
                    )
                    results["training_curve"] = {
                        "train_sizes": train_sizes.tolist(),
                        "train_scores": np.mean(train_scores, axis=1).tolist(),
                        "validation_scores": np.mean(val_scores, axis=1).tolist()
                    }
                except:
                    # Fallback if learning_curve fails
                    pass
        
        elif task_type == "clustering":
            # Clustering
            cluster_labels = model.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            results = {
                "silhouette_score": float(silhouette_avg),
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": model.cluster_centers_.tolist() if hasattr(model, 'cluster_centers_') else [],
                "inertia": float(model.inertia_) if hasattr(model, 'inertia_') else None
            }
        
        return {
            "success": True,
            "results": results,
            "task_type": task_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
