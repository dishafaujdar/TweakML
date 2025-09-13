import { Select, SelectItem, SelectValue } from './ui/select';
import { Dataset } from '../types/api';

interface DatasetSelectorProps {
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  onDatasetSelect: (dataset: Dataset) => void;
}

const DatasetSelector = ({ datasets, selectedDataset, onDatasetSelect }: DatasetSelectorProps) => {
  const handleSelect = (value: string) => {
    const dataset = datasets.find(d => d.name === value);
    if (dataset) {
      onDatasetSelect(dataset);
    }
  };

  const getDatasetDisplayName = (dataset: Dataset) => {
    const displayNames: Record<string, string> = {
      'iris': '🌸 Iris (Classification)',
      'wine': '🍷 Wine Quality (Classification)', 
      'titanic': '🚢 Titanic (Binary Classification)',
      'california_housing': '🏠 California Housing (Regression)',
      'digits': '✏️ Handwritten Digits (Classification)'
    };
    return displayNames[dataset.name] || dataset.name;
  };

  return (
    <div className="space-y-4">
      <Select value={selectedDataset?.name || ''} onValueChange={handleSelect}>
        <SelectValue placeholder="Choose a dataset..." />
        {datasets.map((dataset) => (
          <SelectItem key={dataset.name} value={dataset.name}>
            {getDatasetDisplayName(dataset)}
          </SelectItem>
        ))}
      </Select>

      {selectedDataset && (
        <div className="space-y-3">
          <div className="text-sm space-y-1">
            <p><strong>Shape:</strong> {selectedDataset.shape[0]} rows × {selectedDataset.shape[1]} cols</p>
            <p><strong>Task:</strong> {selectedDataset.task_type}</p>
            <p><strong>Features:</strong> {selectedDataset.features.length}</p>
            <p><strong>Target:</strong> {selectedDataset.target}</p>
          </div>

          <div className="border rounded-md overflow-hidden">
            <div className="bg-gray-50 px-3 py-2 border-b">
              <p className="text-xs font-medium text-gray-700">Dataset Preview (First 5 rows)</p>
            </div>
            <div className="overflow-x-auto max-h-64">
              <table className="w-full text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(selectedDataset.preview[0] || {}).map((key) => (
                      <th key={key} className="px-2 py-1 text-left font-medium text-gray-700 border-r">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selectedDataset.preview.slice(0, 5).map((row, idx) => (
                    <tr key={idx} className="border-t">
                      {Object.values(row).map((value, colIdx) => (
                        <td key={colIdx} className="px-2 py-1 border-r">
                          {typeof value === 'number' ? value.toFixed(2) : String(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetSelector;
