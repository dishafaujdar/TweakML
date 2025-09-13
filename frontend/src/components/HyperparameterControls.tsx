import { Slider } from './ui/slider';
import { Input } from './ui/input';
import { Select, SelectItem, SelectValue } from './ui/select';
import { Model } from '../types/api';

interface HyperparameterControlsProps {
  modelConfig?: Model;
  hyperparameters: Record<string, any>;
  onHyperparameterChange: (params: Record<string, any>) => void;
  disabled?: boolean;
}

const HyperparameterControls = ({ 
  modelConfig, 
  hyperparameters, 
  onHyperparameterChange, 
  disabled 
}: HyperparameterControlsProps) => {
  if (disabled || !modelConfig) {
    return (
      <div className="text-sm text-gray-500 italic">
        Select a model to configure hyperparameters
      </div>
    );
  }

  const handleParamChange = (paramName: string, value: any) => {
    const newParams = { ...hyperparameters, [paramName]: value };
    onHyperparameterChange(newParams);
  };

  const formatParamName = (name: string) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="space-y-4">
      {Object.entries(modelConfig.hyperparameters).map(([paramName, paramConfig]) => {
        const currentValue = hyperparameters[paramName] ?? paramConfig.default;

        return (
          <div key={paramName} className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              {formatParamName(paramName)}
            </label>

            {paramConfig.type === 'int' && (
              <div className="space-y-2">
                <Slider
                  value={[currentValue]}
                  onValueChange={(value: number[]) => handleParamChange(paramName, value[0])}
                  min={paramConfig.min}
                  max={paramConfig.max}
                  step={1}
                  className="w-full"
                />
                <div className="flex items-center space-x-2">
                  <Input
                    type="number"
                    value={currentValue}
                    onChange={(e) => handleParamChange(paramName, parseInt(e.target.value))}
                    min={paramConfig.min}
                    max={paramConfig.max}
                    className="w-20 text-center"
                  />
                  <span className="text-xs text-gray-500">
                    ({paramConfig.min} - {paramConfig.max})
                  </span>
                </div>
              </div>
            )}

            {paramConfig.type === 'float' && (
              <div className="space-y-2">
                <Slider
                  value={[currentValue]}
                  onValueChange={(value: number[]) => handleParamChange(paramName, value[0])}
                  min={paramConfig.min}
                  max={paramConfig.max}
                  step={0.001}
                  className="w-full"
                />
                <div className="flex items-center space-x-2">
                  <Input
                    type="number"
                    value={currentValue}
                    onChange={(e) => handleParamChange(paramName, parseFloat(e.target.value))}
                    min={paramConfig.min}
                    max={paramConfig.max}
                    step={0.001}
                    className="w-24 text-center"
                  />
                  <span className="text-xs text-gray-500">
                    ({paramConfig.min} - {paramConfig.max})
                  </span>
                </div>
              </div>
            )}

            {paramConfig.type === 'boolean' && (
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={currentValue}
                  onChange={(e) => handleParamChange(paramName, e.target.checked)}
                  className="w-4 h-4 text-primary border-gray-300 rounded focus:ring-primary"
                />
                <span className="text-sm text-gray-600">
                  {currentValue ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            )}

            {paramConfig.type === 'select' && (
              <Select 
                value={currentValue} 
                onValueChange={(value: string) => handleParamChange(paramName, value)}
              >
                <SelectValue />
                {paramConfig.options?.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </Select>
            )}

            <div className="text-xs text-gray-500">
              Default: {String(paramConfig.default)}
            </div>
          </div>
        );
      })}

      {Object.keys(modelConfig.hyperparameters).length === 0 && (
        <div className="text-sm text-gray-500 italic">
          This model has no configurable hyperparameters.
        </div>
      )}
    </div>
  );
};

export default HyperparameterControls;
