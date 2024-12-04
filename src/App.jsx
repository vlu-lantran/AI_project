import React from 'react';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          {/* Tham chiếu logo từ thư mục public */}
          <img 
            src={`${process.env.PUBLIC_URL}/Cam.png`} 
            alt="App Logo" 
            style={{ width: '200px', height: 'auto', objectFit: 'contain' }} 
          />
        </div>
      </header>
      <div className="main-content">
        <h3>Welcome to the App</h3>
        <p>This is a sample app to demonstrate logo display.</p>
      </div>
    </div>
  );
}

export default App;
