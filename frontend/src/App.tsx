import { useState } from 'react';
import LandingPage from './components/LandingPage';
import Playground from './components/PlaygroundMain';

function App() {
  const [userEmail, setUserEmail] = useState<string | null>(null);

  const handleEmailSubmit = (email: string) => {
    setUserEmail(email);
  };

  if (!userEmail) {
    return <LandingPage onEmailSubmit={handleEmailSubmit} />;
  }

  return <Playground userEmail={userEmail} />;
}

export default App;
