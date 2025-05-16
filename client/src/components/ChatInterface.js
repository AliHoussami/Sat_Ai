import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './ChatInterface.css';
import MessageItem from './MessageItem';

// Define helper functions
const isMathSolution = (text) => {
  return /(\$\$|\\\[|\\\(|\\frac|\\boxed|\\begin\{.*\})/.test(text) || 
         text.includes('Solve the equation:') || 
         text.includes('system of equations') ||
         (text.includes('Step 1:') && text.includes('Step 2:'));
};

const prepareMathText = (text) => {
  return text; 
};

const formatMathSolutionInternal = (text) => {
   const parts = text.split('\n\n');
   
   return (
     <div className="math-solution">
       {parts.map((part, index) => {
          const preparedPart = prepareMathText(part); 
          if (part.startsWith('Solve the equation:') || part.startsWith('To solve the system of equations:')) {
            const lines = part.split('\n');
            const header = lines[0];
            const restOfContent = lines.slice(1).join('\n');
            return ( <div key={index}> <div className="solution-header">{header}</div> <div className="centered-equation" dangerouslySetInnerHTML={{ __html: prepareMathText(restOfContent) }} /> </div> );
          } else if (part.startsWith('Step ') || part.includes('Equation')) {
             const lines = part.split('\n');
             const step = lines[0];
             const restOfContent = lines.slice(1).join('\n');
             return ( <div key={index}> <div className="solution-step">{step}</div> <div className="centered-equation" dangerouslySetInnerHTML={{ __html: prepareMathText(restOfContent) }} /> </div> );
          } else if (part.startsWith('Final Answer:')) {
             const lines = part.split('\n');
             const header = lines[0];
             const answer = lines.slice(1).join('\n');
             return ( <div key={index}> <div className="solution-final">{header}</div> <div className="final-answer"> <span className="answer-box" dangerouslySetInnerHTML={{ __html: prepareMathText(answer) }} /> </div> </div> );
          } else {
            return <div key={index} dangerouslySetInnerHTML={{ __html: preparedPart }} />;
          }
       })}
     </div>
   );
};

function ChatInterface({ user }) {
  // State
  const [messages, setMessages] = useState([
    { 
      sender: 'bot', 
      text: 'Hello! I\'m your SAT prep assistant. How can I help you today?' 
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(true); // Default to dark mode
  
  // Updated recentChats to start empty, we'll load from the database
  const [recentChats, setRecentChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(100); // Initial zoom at 100%
  
  // Add a state to track chat message history
  const [chatHistory, setChatHistory] = useState({});
  
  // Refs
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const appContainerRef = useRef(null);

  // Load user's chat history on component mount
  useEffect(() => {
    const loadUserChats = async () => {
      if (user?.id) {
        try {
          const response = await axios.get(`http://127.0.0.1:5000/user_questions/${user.id}`);
          
          if (response.data && response.data.questions) {
            // Group questions by timestamp to create chat sessions
            const chatSessions = {};
            const questions = response.data.questions;
            
            // If no questions, we'll just keep the default state
            if (questions.length === 0) {
              return;
            }
            
            // Group by date (rough chat sessions)
            questions.forEach(question => {
              // Create a date string for the day this question was asked
              const date = new Date(question.timestamp);
              const dateStr = date.toLocaleDateString('en-US');
              
              // Use the date as a grouping key
              if (!chatSessions[dateStr]) {
                chatSessions[dateStr] = {
                  id: question.id, // Use first question's ID as chat ID
                  title: question.question_text.substring(0, 30) + '...', // Use first question as title
                  date: dateStr,
                  questions: []
                };
              }
              
              // Add question to this chat session
              chatSessions[dateStr].questions.push({
                id: question.id,
                question: question.question_text,
                answer: question.rag_response,
                timestamp: question.timestamp
              });
            });
            
            // Convert to array and sort by date (newest first)
            const chatsArray = Object.values(chatSessions).sort((a, b) => 
              new Date(b.date) - new Date(a.date)
            );
            
            setRecentChats(chatsArray);
            
            // If we have chats, set the active chat to the most recent
            if (chatsArray.length > 0) {
              setActiveChatId(chatsArray[0].id);
              
              // Load messages for this chat
              const messages = [];
              chatsArray[0].questions.forEach(q => {
                messages.push({ sender: 'user', text: q.question });
                messages.push({ sender: 'bot', text: q.answer });
              });
              
              // If we have messages, set them (otherwise keep default greeting)
              if (messages.length > 0) {
                setMessages(messages);
              }
              
              // Save all chat histories to state
              const historyObj = {};
              chatsArray.forEach(chat => {
                const chatMessages = [];
                chat.questions.forEach(q => {
                  chatMessages.push({ sender: 'user', text: q.question });
                  chatMessages.push({ sender: 'bot', text: q.answer });
                });
                historyObj[chat.id] = chatMessages;
              });
              setChatHistory(historyObj);
            }
          }
        } catch (error) {
          console.error("Error loading chat history:", error);
          // If there's an error, we'll just use the default static data
          const defaultChats = [
            { id: 1, title: 'SAT Math Practice', date: 'May 9, 2025' },
            { id: 2, title: 'Reading Comprehension', date: 'May 8, 2025' },
            { id: 3, title: 'Grammar Review', date: 'May 5, 2025' }
          ];
          setRecentChats(defaultChats);
          setActiveChatId(1);
        }
      } else {
        // No user, use default static data
        const defaultChats = [
          { id: 1, title: 'SAT Math Practice', date: 'May 9, 2025' },
          { id: 2, title: 'Reading Comprehension', date: 'May 8, 2025' },
          { id: 3, title: 'Grammar Review', date: 'May 5, 2025' }
        ];
        setRecentChats(defaultChats);
        setActiveChatId(1);
      }
    };
    
    loadUserChats();
  }, [user?.id]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Render MathJax after any updates to messages
  useEffect(() => {
    const timerId = setTimeout(() => {
      const container = messagesContainerRef.current; 
      if (container && window.MathJax) {
        if (window.MathJax.typesetPromise) {
          window.MathJax.typesetPromise([container])
            .catch(err => console.error('MathJax typesetting failed:', err));
        } else if (window.MathJax.typeset) {
          window.MathJax.typeset([container]); 
        }
      }
    }, 0); 
    return () => clearTimeout(timerId);
  }, [messages]);
  
  // Zoom control with keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Detect Ctrl + Plus (zoom in)
      if (e.ctrlKey && (e.key === '+' || e.key === '=')) {
        e.preventDefault();
        setZoomLevel(prevZoom => Math.min(prevZoom + 10, 200)); // Max 200%
      }
      // Detect Ctrl + Minus (zoom out)
      else if (e.ctrlKey && e.key === '-') {
        e.preventDefault();
        setZoomLevel(prevZoom => Math.max(prevZoom - 10, 70)); // Min 70%
      }
      // Detect Ctrl + 0 (reset zoom)
      else if (e.ctrlKey && e.key === '0') {
        e.preventDefault();
        setZoomLevel(100); // Reset to 100%
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Apply zoom level effect
  useEffect(() => {
    if (appContainerRef.current) {
      appContainerRef.current.style.transform = `scale(${zoomLevel / 100})`;
      appContainerRef.current.style.transformOrigin = 'center top';
    }
  }, [zoomLevel]);

  // Send message handler (updated to store questions in database)
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prevMessages => [...prevMessages, userMessage]);

    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      // Determine question type based on content (simple heuristic)
      let questionType = 'general';
      
      if (/equation|math|algebra|geometry|calculus|formula|solve for|factor|simplify/i.test(currentInput)) {
        questionType = 'math';
      } else if (/reading|passage|text|author|character|theme|literature|comprehension/i.test(currentInput)) {
        questionType = 'reading_writing';
      } else if (/score|scoring|points|test results|percentile|college board/i.test(currentInput)) {
        questionType = 'scoring';
      }
      
      const response = await axios.post('http://127.0.0.1:5000/ask', { 
        question: currentInput,
        user_id: user?.id || '1',
        question_type: questionType
      });
      
      const botResponse = { sender: 'bot', text: response.data.answer };
      
      setMessages(prevMessages => [
        ...prevMessages,
        botResponse
      ]);
      
      // Update chat history for current active chat
      if (activeChatId) {
        setChatHistory(prev => {
          const updatedHistory = { ...prev };
          if (!updatedHistory[activeChatId]) {
            updatedHistory[activeChatId] = [];
          }
          updatedHistory[activeChatId] = [
            ...updatedHistory[activeChatId],
            userMessage,
            botResponse
          ];
          return updatedHistory;
        });
        
        // Update this chat's title if it's a new chat with only the welcome message
        if (messages.length === 1 && messages[0].sender === 'bot' && messages[0].text.includes("Hello! I'm your SAT prep assistant")) {
          setRecentChats(prev => {
            return prev.map(chat => {
              if (chat.id === activeChatId) {
                // Create a title from the user's first question
                const title = currentInput.length > 30 
                  ? currentInput.substring(0, 30) + '...' 
                  : currentInput;
                  
                return { ...chat, title };
              }
              return chat;
            });
          });
        }
      } else {
        // If this is a new conversation (no active chat), create one
        handleNewChat(currentInput, response.data.answer);
      }
      
    } catch (error) {
      console.error("API Error:", error);
      setMessages(prevMessages => [
        ...prevMessages,
        { 
          sender: 'bot', 
          text: 'Sorry, I encountered an error communicating with the assistant. Please try again.' 
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Updated new chat handler
  const handleNewChat = (firstQuestion = null, firstAnswer = null) => {
    const now = new Date();
    const newChatId = Math.max(...(recentChats.length ? recentChats.map(chat => chat.id) : [0])) + 1;
    
    // Create title - either from first question or default
    const title = firstQuestion 
      ? (firstQuestion.length > 30 ? firstQuestion.substring(0, 30) + '...' : firstQuestion)
      : `New Conversation (${now.toLocaleString()})`;
    
    const newChat = {
      id: newChatId,
      title: title,
      date: now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      questions: []
    };
    
    // If we have a first question/answer, add them to chat history
    if (firstQuestion && firstAnswer) {
      newChat.questions.push({
        question: firstQuestion,
        answer: firstAnswer,
        timestamp: now.toISOString()
      });
      
      // Also update chatHistory state
      setChatHistory(prev => ({
        ...prev,
        [newChatId]: [
          { sender: 'user', text: firstQuestion },
          { sender: 'bot', text: firstAnswer }
        ]
      }));
      
      // Set messages for this new chat
      setMessages([
        { sender: 'user', text: firstQuestion },
        { sender: 'bot', text: firstAnswer }
      ]);
    } else {
      // Reset messages to just the greeting for a truly new chat
      setMessages([
        { 
          sender: 'bot', 
          text: 'Hello! I\'m your SAT prep assistant. How can I help you today?' 
        }
      ]);
      
      // Initialize empty chat history for this chat
      setChatHistory(prev => ({
        ...prev,
        [newChatId]: []
      }));
    }
    
    setRecentChats([newChat, ...recentChats]);
    setActiveChatId(newChatId);
  };

  // Updated select chat handler
  const handleSelectChat = (chatId) => {
    setActiveChatId(chatId);
    
    // Load this chat's messages from our state
    if (chatHistory[chatId] && chatHistory[chatId].length > 0) {
      setMessages(chatHistory[chatId]);
    } else {
      // If this is a chat from the database without loaded messages
      const chat = recentChats.find(c => c.id === chatId);
      
      if (chat?.questions && chat.questions.length > 0) {
        // Convert questions array to messages format
        const chatMessages = [];
        chat.questions.forEach(q => {
          chatMessages.push({ sender: 'user', text: q.question });
          chatMessages.push({ sender: 'bot', text: q.answer });
        });
        
        setMessages(chatMessages);
        
        // Also update our chat history state
        setChatHistory(prev => ({
          ...prev,
          [chatId]: chatMessages
        }));
      } else {
        // If no messages found, show default
        setMessages([
          { 
            sender: 'bot', 
            text: `Starting a new conversation. How can I help you with SAT prep today?` 
          }
        ]);
      }
    }
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Memoized helpers for MessageItem
  const memoizedIsMathSolution = useCallback(isMathSolution, []);
  const memoizedFormatMathSolution = useCallback(formatMathSolutionInternal, []);

  return (
    <div className="app-container" ref={appContainerRef} style={{ transform: `scale(${zoomLevel / 100})` }}>
      
      {/* Main application area - No top bar */}
      <div className="main-container">
        {/* Sidebar with recent chats */}
        <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'} ${darkMode ? 'dark-mode' : ''}`}>
          <div className="sidebar-header">
            <h3>Recent Chats</h3>
            <button className="new-chat-btn" onClick={() => handleNewChat()}>+ New Chat</button>
          </div>
          <div className="chat-list">
            {recentChats.map(chat => (
              <div 
                key={chat.id} 
                className={`chat-item ${chat.id === activeChatId ? 'active' : ''}`}
                onClick={() => handleSelectChat(chat.id)}
              >
                <div className="chat-title">{chat.title}</div>
                <div className="chat-date">{chat.date}</div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Main chat area */}
        <div className={`main-content ${sidebarOpen ? 'sidebar-open' : ''}`}>
          <div className={`chat-container ${darkMode ? 'dark-mode' : ''}`}>
            <div className="chat-header">
              <div className="header-left">
                <button className="toggle-sidebar" onClick={toggleSidebar}>
                  {sidebarOpen ? '▶' : '◀'}
                </button>
                <h2>SAT Prep Assistant</h2>
              </div>
              <div className="header-right">
                <button 
                  className="theme-toggle" 
                  onClick={() => setDarkMode(!darkMode)}
                >
                  {darkMode ? '☀️ Light' : '🌙 Dark'}
                </button>
              </div>
            </div>
            
            <div className="chat-messages" ref={messagesContainerRef}>
              {messages.map((message, index) => (
                <MessageItem 
                  key={index} 
                  message={message}
                  formatMathSolution={memoizedFormatMathSolution} 
                  isMathSolution={memoizedIsMathSolution}       
                /> 
              ))}
              {loading && (
                <div className="message bot-message">
                  <div className="message-content loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            
            <form className="chat-input-form" onSubmit={handleSendMessage}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything about SAT prep..."
                disabled={loading}
              />
              <button type="submit" disabled={loading || !input.trim()}>
                Send
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;