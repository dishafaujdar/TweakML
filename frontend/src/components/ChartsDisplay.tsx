import { useState, useEffect } from 'react';
import { TrainingResult } from '../types/api';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

// Simple Plot component as fallback
const SimplePlot = ({ data, title }: { data: any, title: string }) => (
  <div className="border rounded p-4 text-center">
    <h4 className="font-medium mb-2">{title}</h4>
    <p className="text-sm text-gray-600">Chart visualization would appear here</p>
    <div className="text-xs text-gray-400 mt-2">
      {JSON.stringify(data).length > 100 
        ? `${JSON.stringify(data).substring(0, 100)}...`
        : JSON.stringify(data)
      }
    </div>
  </div>
);

interface ChartsDisplayProps {
  results: TrainingResult;
}

const ChartsDisplay = ({ results }: ChartsDisplayProps) => {
  const [PlotlyPlot, setPlotlyPlot] = useState<any>(null);

  // Dynamically import Plotly
  useEffect(() => {
    const loadPlotly = async () => {
      try {
        const plotlyModule = await import('react-plotly.js');
        setPlotlyPlot(() => plotlyModule.default);
      } catch (error) {
        console.error('Failed to load Plotly:', error);
      }
    };

    loadPlotly();
  }, []);

  const { task_type, results: metrics } = results;

  const renderLearningCurve = () => {
    if (!metrics.training_curve) return null;

    const { train_sizes, train_scores, validation_scores } = metrics.training_curve;
    
    if (PlotlyPlot) {
      const data = [
        {
          x: train_sizes,
          y: train_scores,
          type: 'scatter',
          mode: 'markers',
          name: 'Training Score',
          marker: { 
            color: '#3B82F6', 
            size: 10,
            symbol: 'circle'
          }
        },
        {
          x: train_sizes,
          y: validation_scores,
          type: 'scatter',
          mode: 'markers',
          name: 'Validation Score',
          marker: { 
            color: '#EF4444', 
            size: 10,
            symbol: 'diamond'
          }
        }
      ];

      const layout = {
        title: 'Learning Curve - Model Performance vs Training Set Size',
        xaxis: { title: 'Training Set Size' },
        yaxis: { 
          title: task_type === 'classification' ? 'Accuracy' : 'R² Score',
          range: [0, 1]
        },
        width: 600,
        height: 400,
        margin: { l: 60, r: 50, t: 80, b: 60 },
        showlegend: true,
        legend: { x: 0.7, y: 0.2 }
      };

      return <PlotlyPlot data={data} layout={layout} config={{ responsive: true }} />;
    }

    return <SimplePlot data={metrics.training_curve} title="Learning Curve" />;
  };

  const renderDecisionBoundary = () => {
    if (!metrics.decision_boundary) return null;

    const { x_range, y_range, predictions, feature_names } = metrics.decision_boundary;
    
    if (PlotlyPlot) {
      const data = [{
        x: x_range,
        y: y_range,
        z: predictions,
        type: 'contour',
        colorscale: 'Viridis',
        showscale: true,
        contours: {
          coloring: 'fill'
        }
      }];

      const layout = {
        title: 'Decision Boundary - Model Classification Regions',
        xaxis: { 
          title: feature_names[0] || 'Feature 1'
        },
        yaxis: { 
          title: feature_names[1] || 'Feature 2'
        },
        width: 600,
        height: 500,
        margin: { l: 80, r: 50, t: 80, b: 80 }
      };

      return <PlotlyPlot data={data} layout={layout} config={{ responsive: true }} />;
    }

    return <SimplePlot data={metrics.decision_boundary} title="Decision Boundary" />;
  };

  const renderClusterVisualization = () => {
    if (!metrics.cluster_centers) return null;

    const centers = metrics.cluster_centers;
    
    if (PlotlyPlot) {
      const data = centers.map((center, index) => ({
        x: center.map((_, featureIndex) => featureIndex),
        y: center,
        type: 'scatter',
        mode: 'lines+markers',
        name: `Cluster ${index}`,
        line: { width: 2 }
      }));

      const layout = {
        title: 'Cluster Centers',
        xaxis: { title: 'Feature Index' },
        yaxis: { title: 'Feature Value' },
        height: 400,
        margin: { l: 60, r: 50, t: 60, b: 100 }
      };

      return <PlotlyPlot data={data} layout={layout} config={{ responsive: true }} />;
    }

    return <SimplePlot data={centers} title="Cluster Centers" />;
  };

  return (
    <div className="space-y-6">
      {/* Learning Curve - Available for both classification and regression */}
      {metrics.training_curve && (
        <Card>
          <CardHeader>
            <CardTitle>Learning Curve</CardTitle>
          </CardHeader>
          <CardContent>
            {renderLearningCurve()}
          </CardContent>
        </Card>
      )}

      {/* Decision Boundary - Only for classification */}
      {task_type === 'classification' && metrics.decision_boundary && (
        <Card>
          <CardHeader>
            <CardTitle>Decision Boundary</CardTitle>
          </CardHeader>
          <CardContent>
            {renderDecisionBoundary()}
          </CardContent>
        </Card>
      )}

      {/* Clustering Charts */}
      {task_type === 'clustering' && (
        <div className="grid grid-cols-1 gap-6">
          {metrics.cluster_centers && (
            <Card>
              <CardHeader>
                <CardTitle>Cluster Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {renderClusterVisualization()}
                  
                  {metrics.cluster_labels && (
                    <div className="text-sm space-y-2">
                      <h4 className="font-medium">Cluster Summary:</h4>
                      {Array.from(new Set(metrics.cluster_labels)).map(clusterId => {
                        const count = metrics.cluster_labels?.filter(label => label === clusterId).length || 0;
                        const percentage = ((count / (metrics.cluster_labels?.length || 1)) * 100).toFixed(1);
                        return (
                          <div key={clusterId} className="flex justify-between">
                            <span>Cluster {clusterId}:</span>
                            <span>{count} samples ({percentage}%)</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Fallback message */}
      {!metrics.training_curve && 
       !metrics.decision_boundary && 
       !metrics.cluster_centers && (
        <Card>
          <CardContent className="text-center py-8">
            <p className="text-gray-500">No visualizations available for this model configuration.</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ChartsDisplay;
