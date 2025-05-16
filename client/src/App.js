import React, { useState, useEffect } from 'react';
import './App.css';
import Login from './components/Login';
import MainMenu from './components/MainMenu';
import ChatInterface from './components/ChatInterface';
import Navbar from './components/Navbar';
import PracticeTest from './components/PracticeTest';


function App() {
  const [user, setUser] = useState(null);
  const [view, setView] = useState('login'); // 'login', 'dashboard', 'chat', 'practice'
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Check if user is already logged in (could use localStorage)
    const savedUser = localStorage.getItem('sat_prep_user');
    const savedTheme = localStorage.getItem('sat_prep_theme');
    
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      setView('mainmenu');
    }
    
    if (savedTheme === 'dark') {
      setDarkMode(true);
      document.body.classList.add('dark-mode');
    }
  }, []);

  const handleLogin = (userData) => {
    setUser(userData);
    localStorage.setItem('sat_prep_user', JSON.stringify(userData));
    setView('mainmenu');
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('sat_prep_user');
    setView('login');
  };

  const toggleTheme = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    if (newMode) {
      document.body.classList.add('dark-mode');
      localStorage.setItem('sat_prep_theme', 'dark');
    } else {
      document.body.classList.remove('dark-mode');
      localStorage.setItem('sat_prep_theme', 'light');
    }
  };

  const navigateTo = (newView) => {
    setView(newView);
  };

  return (
    <div className={`App ${darkMode ? 'dark-mode' : ''}`}>
      {user && (
        <Navbar 
          user={user} 
          currentView={view} 
          onNavigate={navigateTo} 
          onLogout={handleLogout} 
          darkMode={darkMode}
          onToggleTheme={toggleTheme}
        />
      )}
      
      <main className="main-content">
        {view === 'login' && <Login onLogin={handleLogin} darkMode={darkMode} />}
        {view === 'mainmenu' && <MainMenu user={user} darkMode={darkMode} onNavigate={navigateTo} />}
        {view === 'chat' && <ChatInterface user={user} darkMode={darkMode} />}
        {view === 'practice' && <PracticeTest user={user} darkMode={darkMode} />}
      </main>
    </div>
  );
}

export default App;