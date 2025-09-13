import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

interface LandingPageProps {
  onEmailSubmit: (email: string) => void;
}

const LandingPage = ({ onEmailSubmit }: LandingPageProps) => {
  const [email, setEmail] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email.trim()) {
      onEmailSubmit(email.trim());
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-md w-full space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🧠 ML Playground
          </h1>
          <p className="text-lg text-gray-600">
            Learn and experiment with machine learning models
          </p>
        </div>

        {/* Email Form */}
        <Card className="shadow-lg">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Get Started</CardTitle>
            <CardDescription>
              Enter your email to access the interactive ML playground
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Input
                  type="email"
                  placeholder="Enter your email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="text-center"
                  required
                />
              </div>
              <Button type="submit" className="w-full">
                Enter Playground
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Features */}
        <div className="grid grid-cols-1 gap-4 text-center">
          <div className="bg-white/50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-800">📊 5 Datasets</h3>
            <p className="text-sm text-gray-600">Iris, Wine, Titanic, Housing, Digits</p>
          </div>
          <div className="bg-white/50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-800">🤖 Multiple Models</h3>
            <p className="text-sm text-gray-600">Decision Trees, Random Forest, SVM, and more</p>
          </div>
          <div className="bg-white/50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-800">📈 Interactive Charts</h3>
            <p className="text-sm text-gray-600">Real-time visualizations and metrics</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
