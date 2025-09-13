// API types
export interface Dataset {
  name: string;
  shape: number[];
  features: string[];
  target: string;
  task_type: string;
  preview: Record<string, any>[];
}

export interface Model {
  name: string;
  hyperparameters: Record<string, HyperParameter>;
}

export interface HyperParameter {
  type: 'int' | 'float' | 'boolean' | 'select';
  min?: number;
  max?: number;
  default: any;
  options?: string[];
}

export interface TrainingRequest {
  dataset_name: string;
  model_type: string;
  hyperparameters: Record<string, any>;
}

export interface TrainingResult {
  success: boolean;
  results: {
    // Classification metrics
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    
    // Regression metrics
    rmse?: number;
    mae?: number;
    r2_score?: number;
    
    // Clustering metrics
    silhouette_score?: number;
    cluster_labels?: number[];
    cluster_centers?: number[][];
    inertia?: number;
    
    // Model fitting data for curves
    training_curve?: {
      train_sizes: number[];
      train_scores: number[];
      validation_scores: number[];
    };
    
    // Decision boundary data (for 2D visualization)
    decision_boundary?: {
      x_range: number[];
      y_range: number[];
      predictions: number[][];
      feature_names: [string, string];
    };
  };
  task_type: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}
