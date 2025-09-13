import { TrainingResult } from '../types/api';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface MetricsDisplayProps {
  results: TrainingResult;
}

const MetricsDisplay = ({ results }: MetricsDisplayProps) => {
  const { task_type, results: metrics } = results;

  const formatNumber = (num: number) => {
    return Number(num).toFixed(3);
  };

  const getMetricColor = (value: number, isHigherBetter: boolean = true) => {
    const threshold = isHigherBetter ? 0.8 : 0.2;
    if (isHigherBetter) {
      return value >= threshold ? 'text-green-600' : value >= 0.6 ? 'text-yellow-600' : 'text-red-600';
    } else {
      return value <= threshold ? 'text-green-600' : value <= 0.4 ? 'text-yellow-600' : 'text-red-600';
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {task_type === 'classification' && (
        <>
          {metrics.accuracy !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">Accuracy</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.accuracy)}`}>
                  {formatNumber(metrics.accuracy)}
                </div>
                <div className="text-xs text-gray-500">
                  {(metrics.accuracy * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
          )}

          {metrics.precision !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">Precision</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.precision)}`}>
                  {formatNumber(metrics.precision)}
                </div>
                <div className="text-xs text-gray-500">
                  {(metrics.precision * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
          )}

          {metrics.recall !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">Recall</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.recall)}`}>
                  {formatNumber(metrics.recall)}
                </div>
                <div className="text-xs text-gray-500">
                  {(metrics.recall * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
          )}

          {metrics.f1_score !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">F1 Score</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.f1_score)}`}>
                  {formatNumber(metrics.f1_score)}
                </div>
                <div className="text-xs text-gray-500">
                  {(metrics.f1_score * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {task_type === 'regression' && (
        <>
          {metrics.r2_score !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">R² Score</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.r2_score)}`}>
                  {formatNumber(metrics.r2_score)}
                </div>
                <div className="text-xs text-gray-500">
                  Coefficient of determination
                </div>
              </CardContent>
            </Card>
          )}

          {metrics.rmse !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">RMSE</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.rmse, false)}`}>
                  {formatNumber(metrics.rmse)}
                </div>
                <div className="text-xs text-gray-500">
                  Root Mean Squared Error
                </div>
              </CardContent>
            </Card>
          )}

          {metrics.mae !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">MAE</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.mae, false)}`}>
                  {formatNumber(metrics.mae)}
                </div>
                <div className="text-xs text-gray-500">
                  Mean Absolute Error
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {task_type === 'clustering' && (
        <>
          {metrics.silhouette_score !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">Silhouette Score</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.silhouette_score)}`}>
                  {formatNumber(metrics.silhouette_score)}
                </div>
                <div className="text-xs text-gray-500">
                  Cluster quality measure
                </div>
              </CardContent>
            </Card>
          )}

          {metrics.inertia !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-700">Inertia</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className={`text-2xl font-bold ${getMetricColor(metrics.inertia, false)}`}>
                  {formatNumber(metrics.inertia)}
                </div>
                <div className="text-xs text-gray-500">
                  Within-cluster sum of squares
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
};

export default MetricsDisplay;
