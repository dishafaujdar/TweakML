import React from 'react';
import { TrainingResult } from '../types/api';
import { Card, CardContent } from '@/components/ui/card';
import MetricsDisplay from './MetricsDisplay';
import ChartsDisplay from './ChartsDisplay';

interface Props {
  results: TrainingResult | null;
  isLoading: boolean;
}

const ResultsView: React.FC<Props> = ({ results, isLoading }) => {
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="space-y-6">
        <Card>
          <CardContent className="p-8">
            <div className="text-center text-gray-500">
              <p>Adjust hyperparameters above to train your model and see results here.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <MetricsDisplay results={results} />
      <ChartsDisplay results={results} />
    </div>
  );
};

export default ResultsView;
