import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import DatasetSelector from './DatasetSelector';
import ModelSelector from './ModelSelector';
import HyperparameterControls from './HyperparameterControls';
import ResultsView from './ResultsView';
import { apiClient } from '../lib/api';
import { Dataset, Model, TrainingResult } from '../types/api';

interface PlaygroundProps {
  userEmail: string;
}

const Playground = ({ userEmail }: PlaygroundProps) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<Record<string, Model>>({});
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [trainingResults, setTrainingResults] = useState<TrainingResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        const [datasetsResponse, modelsResponse] = await Promise.all([
          apiClient.getDatasets(),
          apiClient.getModels()
        ]);
        
        setDatasets(datasetsResponse as Dataset[]);
        setModels(modelsResponse as Record<string, Model>);
      } catch (err) {
        setError('Failed to load data from server');
        console.error('Error loading data:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, []);

  // Update available models when dataset changes
  const availableModels: Record<string, Model> = selectedDataset && typeof models[selectedDataset.task_type] === 'object'
    ? (models[selectedDataset.task_type] as unknown as Record<string, Model>)
    : {};

  // Reset model selection when dataset changes
  useEffect(() => {
    if (selectedDataset && !availableModels[selectedModel]) {
      setSelectedModel('');
      setHyperparameters({});
      setTrainingResults(null);
    }
  }, [selectedDataset, availableModels, selectedModel]);

  // Initialize hyperparameters when model changes
  useEffect(() => {
    if (selectedModel && availableModels[selectedModel]) {
      const modelConfig = availableModels[selectedModel];
      const defaultParams: Record<string, any> = {};
      
      Object.entries(modelConfig.hyperparameters).forEach(([paramName, paramConfig]) => {
        defaultParams[paramName] = paramConfig.default;
      });
      
      setHyperparameters(defaultParams);
      setTrainingResults(null);
    }
  }, [selectedModel, availableModels]);

  const handleTrain = async () => {
    if (!selectedDataset || !selectedModel) {
      setError('Please select both a dataset and model');
      return;
    }

    try {
      setIsTraining(true);
      setError(null);
      
      const response = await apiClient.trainModel({
        dataset_name: selectedDataset.name,
        model_type: selectedModel,
        hyperparameters
      });
      
      setTrainingResults(response as TrainingResult);
    } catch (err) {
      setError('Failed to train model');
      console.error('Training error:', err);
    } finally {
      setIsTraining(false);
    }
  };

  const canTrain = selectedDataset && selectedModel && !isTraining;
  const selectedModelConfig = selectedModel ? availableModels[selectedModel] : undefined;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading ML Playground...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">🧠 ML Playground</h1>
              <p className="text-sm text-gray-600">Welcome, {userEmail}</p>
            </div>
            <div className="flex items-center space-x-4">
              <Button
                onClick={handleTrain}
                disabled={!canTrain}
                variant={canTrain ? 'default' : 'outline'}
                size="lg"
              >
                {isTraining ? 'Training...' : 'Train Model'}
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sidebar - Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Dataset Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">1. Select Dataset</CardTitle>
                <CardDescription>
                  Choose from {datasets.length} available datasets
                </CardDescription>
              </CardHeader>
              <CardContent>
                <DatasetSelector
                  datasets={datasets}
                  selectedDataset={selectedDataset}
                  onDatasetSelect={setSelectedDataset}
                />
              </CardContent>
            </Card>

            {/* Model Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">2. Choose Model</CardTitle>
                <CardDescription>
                  Pick a model for {selectedDataset?.task_type || 'your task'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ModelSelector
                  models={availableModels}
                  selectedModel={selectedModel}
                  onModelSelect={setSelectedModel}
                  disabled={!selectedDataset}
                />
              </CardContent>
            </Card>

            {/* Hyperparameter Controls */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">3. Tune Parameters</CardTitle>
                <CardDescription>
                  Adjust hyperparameters for optimal performance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <HyperparameterControls
                  modelConfig={selectedModelConfig}
                  hyperparameters={hyperparameters}
                  onHyperparameterChange={setHyperparameters}
                  disabled={!selectedModel}
                />
              </CardContent>
            </Card>
          </div>

          {/* Main Content - Results */}
          <div className="lg:col-span-2">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="text-lg">Training Results</CardTitle>
                <CardDescription>
                  Model performance metrics and visualizations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {error && (
                  <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                )}
                
                <ResultsView
                  results={trainingResults}
                  dataset={selectedDataset}
                  modelName={selectedModelConfig?.name || selectedModel}
                  isTraining={isTraining}
                />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-gray-500 text-sm">
            ML Playground - Learn machine learning through interactive experimentation
          </p>
        </div>
      </div>
    </div>
  );
};

export default Playground;
