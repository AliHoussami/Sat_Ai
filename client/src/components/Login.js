import React, { useState } from 'react';
import axios from 'axios';
import './Login.css';

function Login({ onLogin, darkMode }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    if (!username || !password) {
      setError('Please enter both username and password');
      setLoading(false);
      return;
    }

    try {
      if (isRegistering) {
        // Register new user
        const response = await axios.post('http://127.0.0.1:5000/register', {
          username,
          password
        });
        
        if (response.data.success) {
          // Successfully registered, log the user in
          onLogin({ 
            id: response.data.user_id, 
            username: response.data.username 
          });
        }
      } else {
        // Login with existing account
        const response = await axios.post('http://127.0.0.1:5000/login', {
          username,
          password
        });
        
        if (response.data.success) {
          // Successfully logged in
          onLogin({ 
            id: response.data.user_id, 
            username: response.data.username,
            weakAreas: response.data.weak_areas
          });
        }
      }
    } catch (error) {
      console.error('Auth error:', error);
      setError(error.response?.data?.error || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`login-container ${darkMode ? 'dark-mode' : ''}`}>
      <div className="login-content">
        <div className="login-header">
          <h1>SAT Preparation AI</h1>
          <p>Your intelligent assistant for SAT success</p>
        </div>
        
        <div className="login-form-container">
          <div className="login-form-header">
            <h2>{isRegistering ? 'Create an Account' : 'Welcome Back'}</h2>
            <p>{isRegistering 
              ? 'Register to start your SAT preparation journey' 
              : 'Login to continue your SAT preparation'}</p>
          </div>
          
          {error && <div className="alert alert-error" style={{
    padding: '0.75rem 1rem',
    borderRadius: '4px',
    marginBottom: '1.5rem',
    backgroundColor: '#fff0f0', 
    color: '#d32f2f',
    borderLeft: '4px solid #d32f2f'
  }}>{error}</div>}
          
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input 
                id="username"
                type="text" 
                className="form-control"
                value={username} 
                onChange={(e) => setUsername(e.target.value)} 
                disabled={loading}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input 
                id="password"
                type="password" 
                className="form-control"
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                disabled={loading}
              />
            </div>
            
            <button 
              type="submit" 
              className="btn btn-primary login-button"
              disabled={loading}
            >
              {loading 
                ? 'Please wait...' 
                : isRegistering ? 'Create Account' : 'Login'
              }
            </button>
          </form>
          
          <div className="login-toggle">
            {isRegistering 
              ? 'Already have an account? ' 
              : 'Don\'t have an account? '
            }
            <button 
              className="toggle-button"
              onClick={() => {
                setIsRegistering(!isRegistering);
                setError('');
              }}
              disabled={loading}
            >
              {isRegistering ? 'Login' : 'Register'}
            </button>
          </div>
        </div>
        
        <div className="login-features">
          <h3>Prepare for success with our AI assistant</h3>
          <div className="features-grid">
            <div className="feature-item">
              <div className="feature-icon">📚</div>
              <h4>Smart Practice</h4>
              <p>Get personalized SAT questions based on your performance</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">💬</div>
              <h4>AI Tutor</h4>
              <p>Ask questions and get instant, detailed explanations</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">📊</div>
              <h4>Goal Setting</h4>
              <p>Set targets and stay motivated on your journey to success</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">🎯</div>
              <h4>Focus on Weaknesses</h4>
              <p>Identify and improve your weak areas</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;