import React from 'react';

// You might want to move isMathSolution and formatMathSolution to a separate
// utils.js file and import them here and in ChatInterface, or pass them as props.
// For simplicity here, assuming they are available or passed down.
// Let's assume formatMathSolution and isMathSolution are passed as props for clarity:

const MessageItem = React.memo(({ message, formatMathSolution, isMathSolution }) => {
  // Add a console log to see when it *actually* renders
  // console.log(`Rendering MessageItem: ${message.sender} - ${message.text.substring(0, 20)}...`);

  const renderContent = () => {
    if (message.sender === 'bot' && isMathSolution(message.text)) {
      // Use the formatting function passed as a prop
      return formatMathSolution(message.text);
    } else {
      // Render user messages or non-math bot messages
      // Using dangerouslySetInnerHTML allows MathJax to process potential inline math ($...$)
      // Make sure message.text is appropriately sanitized if needed, although less critical for user's own messages.
      return <div dangerouslySetInnerHTML={{ __html: message.text }}></div>;
    }
  };

  return (
    <div className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
      <div className="message-content">
        {renderContent()}
      </div>
    </div>
  );
}); // Wrap component definition in React.memo

export default MessageItem;