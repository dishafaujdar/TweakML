import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
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
  const [models, setModels] = useState<Record<string, Record<string, Model>>>({});
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [isTraining, setIsTraining] = useState(false);
  const [results, setResults] = useState<TrainingResult | null>(null);
  const [error, setError] = useState<string>('');

  // Load datasets and models on component mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [datasetsResponse, modelsResponse] = await Promise.all([
          apiClient.getDatasets(),
          apiClient.getModels()
        ]);
        
        setDatasets(datasetsResponse as Dataset[]);
        setModels(modelsResponse as Record<string, Record<string, Model>>);
      } catch (err) {
        setError('Failed to load initial data. Make sure the backend is running.');
        console.error('Error loading data:', err);
      }
    };

    loadData();
  }, []);

  // Reset model and hyperparameters when dataset changes
  useEffect(() => {
    if (selectedDataset) {
      setSelectedModel('');
      setHyperparameters({});
      setResults(null);
    }
  }, [selectedDataset]);

  // Reset hyperparameters when model changes
  useEffect(() => {
    if (selectedModel && selectedDataset) {
      const taskType = selectedDataset.task_type;
      const modelConfig = models[taskType]?.[selectedModel];
      
      if (modelConfig) {
        const defaultParams: Record<string, any> = {};
        Object.entries(modelConfig.hyperparameters).forEach(([key, param]) => {
          defaultParams[key] = param.default;
        });
        setHyperparameters(defaultParams);
      }
      setResults(null);
    }
  }, [selectedModel, selectedDataset, models]);

  const trainModel = useCallback(async () => {
    if (!selectedDataset || !selectedModel) {
      return;
    }

    setIsTraining(true);
    setError('');
    
    try {
      const response = await apiClient.trainModel({
        dataset_name: selectedDataset.name,
        model_type: selectedModel,
        hyperparameters: hyperparameters
      });
      
      setResults(response as TrainingResult);
    } catch (err) {
      setError('Training failed. Please check your parameters and try again.');
      console.error('Training error:', err);
    } finally {
      setIsTraining(false);
    }
  }, [selectedDataset, selectedModel, hyperparameters]);

  // Auto-train when hyperparameters change (with debouncing)
  useEffect(() => {
    if (!selectedDataset || !selectedModel || Object.keys(hyperparameters).length === 0) {
      return;
    }

    // Debounce training to avoid too many API calls
    const timeoutId = setTimeout(() => {
      trainModel();
    }, 1000); // Wait 1 second after last hyperparameter change

    return () => clearTimeout(timeoutId);
  }, [selectedDataset, selectedModel, hyperparameters, trainModel]);

  const availableModels = selectedDataset 
    ? models[selectedDataset.task_type] || {}
    : {};

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">🧠 ML Playground</h1>
              <p className="text-sm text-gray-600">Welcome, {userEmail}</p>
            </div>
            <Button 
              onClick={() => window.location.reload()} 
              variant="outline"
              size="sm"
            >
              Reset Session
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <div className="text-red-800">{error}</div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Sidebar - Dataset and Model Selection */}
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>1. Select Dataset</CardTitle>
              </CardHeader>
              <CardContent>
                <DatasetSelector
                  datasets={datasets}
                  selectedDataset={selectedDataset}
                  onDatasetSelect={setSelectedDataset}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>2. Select Model</CardTitle>
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

            <Card>
              <CardHeader>
                <CardTitle>3. Tune Hyperparameters</CardTitle>
                {isTraining && (
                  <div className="flex items-center text-sm text-blue-600">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                    Auto-training...
                  </div>
                )}
              </CardHeader>
              <CardContent>
                <HyperparameterControls
                  modelConfig={availableModels[selectedModel]}
                  hyperparameters={hyperparameters}
                  onHyperparameterChange={setHyperparameters}
                  disabled={!selectedModel}
                />
              </CardContent>
            </Card>

            {/* Info card about auto-training */}
            <Card className="bg-blue-50 border-blue-200">
              <CardContent className="p-4">
                <div className="text-sm text-blue-800">
                  <h4 className="font-medium mb-1">🔄 Auto-Training</h4>
                  <p>The model automatically retrains when you adjust hyperparameters. Changes are applied after 1 second of inactivity.</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Panel - Results */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Results & Learning Curves</CardTitle>
              </CardHeader>
              <CardContent>
                <ResultsView
                  results={results}
                  isLoading={isTraining}
                />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Playground;
