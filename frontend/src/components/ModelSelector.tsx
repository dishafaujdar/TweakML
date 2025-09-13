import { Select, SelectItem, SelectValue } from './ui/select';
import { Model } from '../types/api';

interface ModelSelectorProps {
  models: Record<string, Model>;
  selectedModel: string;
  onModelSelect: (modelType: string) => void;
  disabled?: boolean;
}

const ModelSelector = ({ models, selectedModel, onModelSelect, disabled }: ModelSelectorProps) => {
  const modelEntries = Object.entries(models);

  if (disabled || modelEntries.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        {disabled ? 'Select a dataset first' : 'No models available'}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Select value={selectedModel} onValueChange={onModelSelect}>
        <SelectValue placeholder="Choose a model..." />
        {modelEntries.map(([modelType, model]) => (
          <SelectItem key={modelType} value={modelType}>
            {model.name}
          </SelectItem>
        ))}
      </Select>

      {selectedModel && models[selectedModel] && (
        <div className="text-sm text-gray-600">
          <p><strong>Selected:</strong> {models[selectedModel].name}</p>
          <p><strong>Hyperparameters:</strong> {Object.keys(models[selectedModel].hyperparameters).length}</p>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;
