import React from 'react';
import './MainMenu.css';

function MainMenu({ darkMode, user, onNavigate }) {
  return (
    <div className={`main-menu-container ${darkMode ? 'dark-mode' : ''}`}>
      <div className="hero-section">
        <h1>Welcome to SAT Prep AI</h1>
        <p className="hero-subtitle">Your AI-powered assistant for SAT success</p>
        <div className="hero-buttons">
          <button onClick={() => onNavigate('chat')} className="btn btn-secondary">Start Chatting</button>
        </div>
      </div>
      
      <div className="features-section" style={{textAlign: 'center'}}>
        <h2>How to Use Our SAT Prep Chatbot</h2>
        
        <div className="features-grid" style={{display: 'flex', justifyContent: 'center', gap: '20px'}}>
          <div className="feature-card">
            <div className="feature-icon">💬</div>
            <h3>Ask Questions</h3>
            <p>Ask any SAT-related question and get instant, accurate answers backed by official College Board materials.</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">📝</div>
            <h3>Practice Questions</h3>
            <p>Request practice questions from any SAT section (Math, Reading, or Writing) and test your skills with instant feedback.</p>
          </div>          
        </div>
      </div>
    </div>
  );
}

export default MainMenu;