import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PracticeTest.css';

function PracticeTest({ user, darkMode }) {
  const [testState, setTestState] = useState('intro'); // intro, active, review
  const [testType, setTestType] = useState('');
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [timeLeft, setTimeLeft] = useState(0);
  const [testResults, setTestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Timer effect when test is active
  useEffect(() => {
    let timer;
    if (testState === 'active' && timeLeft > 0) {
      timer = setTimeout(() => {
        setTimeLeft(timeLeft - 1);
      }, 1000);

      // Auto-submit when time expires
      if (timeLeft === 0) {
        handleFinishTest();
      }
    }
    return () => clearTimeout(timer);
  }, [timeLeft, testState]);

  const startTest = async (type) => {
    setLoading(true);
    setError('');
    
    try {
      // Fetch questions for the test
      const response = await axios.get('http://127.0.0.1:5000/questions');
      const fetchedQuestions = response.data.questions || [];
      
      if (fetchedQuestions.length === 0) {
        setError('No questions available for this test');
        setLoading(false);
        return;
      }
      
      setQuestions(fetchedQuestions);
      setTestType(type);
      setCurrentQuestionIndex(0);
      setUserAnswers({});
      
      // Set timer based on test type
      switch (type) {
        case 'quick':
          setTimeLeft(5 * 60); // 5 minutes
          break;
        case 'section':
          setTimeLeft(25 * 60); // 25 minutes
          break;
        case 'full':
          setTimeLeft(65 * 60); // 65 minutes
          break;
        default:
          setTimeLeft(10 * 60); // Default 10 minutes
      }
      
      setTestState('active');
      setLoading(false);
    } catch (err) {
      console.error("Error fetching test questions:", err);
      setError('Failed to load test questions');
      setLoading(false);
    }
  };

  const handleAnswerSelect = (questionId, answer) => {
    setUserAnswers({
      ...userAnswers,
      [questionId]: answer
    });
  };

  const navigateQuestion = (direction) => {
    if (direction === 'next' && currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else if (direction === 'prev' && currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleFinishTest = async () => {
    setLoading(true);
    
    try {
      // Process all answers and check them
      const results = {
        totalQuestions: questions.length,
        attemptedQuestions: Object.keys(userAnswers).length,
        correctAnswers: 0,
        incorrectAnswers: 0,
        unansweredQuestions: 0,
        questionResults: []
      };
      
      // This would normally be handled by the backend
      for (const question of questions) {
        const questionId = question.question_id;
        const userAnswer = userAnswers[questionId] || '';
        const isCorrect = userAnswer.toLowerCase() === question.answer_text.toLowerCase();
        
        if (userAnswer) {
          if (isCorrect) {
            results.correctAnswers++;
          } else {
            results.incorrectAnswers++;
          }
        } else {
          results.unansweredQuestions++;
        }
        
        results.questionResults.push({
          questionId,
          question: question.question_text,
          userAnswer,
          correctAnswer: question.answer_text,
          isCorrect,
          explanation: question.explanation
        });
      }
      
      // In a real app, you'd submit these to the backend
      // const response = await axios.post('http://127.0.0.1:5000/submit_test', {
      //   user_id: user.id,
      //   test_type: testType,
      //   answers: userAnswers,
      //   time_taken: getTestDuration() - timeLeft
      // });
      
      setTestResults(results);
      setTestState('review');
      setLoading(false);
    } catch (err) {
      console.error("Error submitting test:", err);
      setError('Failed to submit test results');
      setLoading(false);
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  const getTestDuration = () => {
    switch (testType) {
      case 'quick': return 5 * 60;
      case 'section': return 25 * 60;
      case 'full': return 65 * 60;
      default: return 10 * 60;
    }
  };

  const renderIntroScreen = () => (
    <div className="test-intro">
      <h2>SAT Practice Tests</h2>
      <p>Choose a practice test format to begin:</p>
      
      <div className="test-options">
        <div className="test-option" onClick={() => startTest('quick')}>
          <h3>Quick Practice</h3>
          <p>5 minutes • 5 questions</p>
          <p className="test-description">A brief practice session to test specific skills</p>
        </div>
        
        <div className="test-option" onClick={() => startTest('section')}>
          <h3>Section Practice</h3>
          <p>25 minutes • 15 questions</p>
          <p className="test-description">Practice a complete section of the SAT</p>
        </div>
        
        <div className="test-option" onClick={() => startTest('full')}>
          <h3>Full Test</h3>
          <p>65 minutes • 40 questions</p>
          <p className="test-description">Take a comprehensive practice test</p>
        </div>
      </div>
    </div>
  );

  const renderActiveTest = () => {
    if (questions.length === 0) return <div>No questions available</div>;
    
    const currentQuestion = questions[currentQuestionIndex];
    
    return (
      <div className="active-test">
        <div className="test-header">
          <div className="test-info">
            <h3>{testType.charAt(0).toUpperCase() + testType.slice(1)} Practice Test</h3>
            <p>Question {currentQuestionIndex + 1} of {questions.length}</p>
          </div>
          
          <div className="test-timer">
            <span className={timeLeft < 60 ? 'timer-warning' : ''}>
              Time remaining: {formatTime(timeLeft)}
            </span>
          </div>
        </div>
        
        <div className="question-container">
          <p className="question-text">{currentQuestion.question_text}</p>
          <p className="question-topic">Topic: {currentQuestion.topic}</p>
          
          <div className="answer-input">
            <label htmlFor="answer">Your Answer:</label>
            <input
              type="text"
              id="answer"
              className="form-control"
              value={userAnswers[currentQuestion.question_id] || ''}
              onChange={(e) => handleAnswerSelect(currentQuestion.question_id, e.target.value)}
            />
          </div>
        </div>
        
        <div className="test-navigation">
          <button 
            className="btn btn-secondary"
            onClick={() => navigateQuestion('prev')}
            disabled={currentQuestionIndex === 0}
          >
            Previous
          </button>
          
          {currentQuestionIndex < questions.length - 1 ? (
            <button 
              className="btn btn-primary"
              onClick={() => navigateQuestion('next')}
            >
              Next
            </button>
          ) : (
            <button 
              className="btn btn-primary finish-button"
              onClick={handleFinishTest}
            >
              Finish Test
            </button>
          )}
        </div>
        
        <div className="question-progress">
          {questions.map((q, index) => (
            <div 
              key={q.question_id}
              className={`progress-dot ${currentQuestionIndex === index ? 'current' : ''} ${userAnswers[q.question_id] ? 'answered' : ''}`}
              onClick={() => setCurrentQuestionIndex(index)}
            />
          ))}
        </div>
      </div>
    );
  };

  const renderTestReview = () => {
    if (!testResults) return <div>No test results available</div>;
    
    return (
      <div className="test-review">
        <div className="review-header">
          <h2>Test Results</h2>
          <p className="test-type">{testType.charAt(0).toUpperCase() + testType.slice(1)} Practice Test</p>
        </div>
        
        <div className="results-summary">
          <div className="summary-item">
            <span className="summary-value">{testResults.totalQuestions}</span>
            <span className="summary-label">Total Questions</span>
          </div>
          <div className="summary-item">
            <span className="summary-value">{testResults.correctAnswers}</span>
            <span className="summary-label">Correct</span>
          </div>
          <div className="summary-item">
            <span className="summary-value">{testResults.incorrectAnswers}</span>
            <span className="summary-label">Incorrect</span>
          </div>
          <div className="summary-item">
            <span className="summary-value">{testResults.unansweredQuestions}</span>
            <span className="summary-label">Unanswered</span>
          </div>
          <div className="summary-item">
            <span className="summary-value">
              {Math.round((testResults.correctAnswers / testResults.totalQuestions) * 100)}%
            </span>
            <span className="summary-label">Score</span>
          </div>
        </div>
        
        <div className="question-reviews">
          <h3>Question Review</h3>
          
          {testResults.questionResults.map((result, index) => (
            <div key={result.questionId} className={`review-item ${result.isCorrect ? 'correct' : result.userAnswer ? 'incorrect' : 'unanswered'}`}>
              <div className="review-question">
                <span className="question-number">Question {index + 1}:</span> {result.question}
              </div>
              
              <div className="review-answers">
                <div className="user-answer">
                  <span className="answer-label">Your answer:</span>
                  {result.userAnswer || <span className="no-answer">No answer provided</span>}
                </div>
                
                {!result.isCorrect && (
                  <div className="correct-answer">
                    <span className="answer-label">Correct answer:</span>
                    {result.correctAnswer}
                  </div>
                )}
              </div>
              
              {!result.isCorrect && result.explanation && (
                <div className="answer-explanation">
                  <span className="explanation-label">Explanation:</span>
                  {result.explanation}
                </div>
              )}
            </div>
          ))}
        </div>
        
        <div className="review-actions">
          <button className="btn btn-secondary" onClick={() => setTestState('intro')}>
            Back to Tests
          </button>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className={`practice-test-container ${darkMode ? 'dark-mode' : ''}`}>
      {error && <div className="alert alert-error">{error}</div>}
      
      {testState === 'intro' && renderIntroScreen()}
      {testState === 'active' && renderActiveTest()}
      {testState === 'review' && renderTestReview()}
    </div>
  );
}

export default PracticeTest;