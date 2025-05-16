import React from 'react';
import './Navbar.css';

function Navbar({ user, currentView, onNavigate, onLogout, darkMode, onToggleTheme }) {
  return (
    <nav className={`navbar ${darkMode ? 'dark-mode' : ''}`}>
      <div className="navbar-container">
        <div className="navbar-brand">
          <img 
            src={darkMode ? "/sat-logo-dark.png" : "/sat-logo-light.png"} 
            alt="SAT Prep" 
            className="navbar-logo"
            onError={(e) => {e.target.src = "https://via.placeholder.com/40"; e.target.onerror = null;}}
          />
          <h1>SAT Prep AI</h1>
        </div>
        
        <div className="navbar-menu">
          <button 
            className={`navbar-item ${currentView === 'mainmenu' ? 'active' : ''}`}
            onClick={() => onNavigate('mainmenu')}
          >
            Main Menu
          </button>
          <button 
            className={`navbar-item ${currentView === 'chat' ? 'active' : ''}`}
            onClick={() => onNavigate('chat')}
          >
            AI Assistant
          </button>
          <button 
            className={`navbar-item ${currentView === 'practice' ? 'active' : ''}`}
            onClick={() => onNavigate('practice')}
          >
            Practice Tests
          </button>
        </div>
        
        <div className="navbar-actions">
          <div className="user-profile">
            <span className="user-name">{user.username}</span>
            <span className="user-avatar">{user.username.charAt(0).toUpperCase()}</span>
          </div>
          
          <button className="theme-toggle" onClick={onToggleTheme}>
            {darkMode ? '☀️' : '🌙'}
          </button>
          
          <button className="logout-button" onClick={onLogout}>
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;