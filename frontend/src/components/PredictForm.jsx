/**
 * PredictForm.jsx — Customer message input form
 *
 * Provides a large textarea for the user to type a customer support message,
 * validates minimum 3 characters, shows a loading spinner during prediction,
 * and displays errors if the request fails.
 */

import { useState } from 'react';

const PredictForm = ({ onPredict, isLoading }) => {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validate minimum 3 characters
    if (message.trim().length < 3) {
      setError('Please enter at least 3 characters.');
      return;
    }

    try {
      await onPredict(message.trim());
      setMessage(''); // Clear input on success
    } catch (err) {
      const errorMsg =
        err.response?.data?.error ||
        err.message ||
        'Failed to get prediction. Please try again.';
      setError(errorMsg);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full" id="predict-form">
      <div className="space-y-4">
        {/* Textarea for customer message */}
        <div>
          <label
            htmlFor="message-input"
            className="block text-sm font-semibold text-gray-700 mb-2"
          >
            Customer Support Message
          </label>
          <textarea
            id="message-input"
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              if (error) setError(''); // Clear error on change
            }}
            placeholder="e.g., I want to cancel my order and get a refund..."
            rows={5}
            className="input-field resize-none text-base leading-relaxed"
            disabled={isLoading}
          />
          <p className="mt-1.5 text-xs text-gray-400">
            Minimum 3 characters required • Type a real customer support message for best results
          </p>
        </div>

        {/* Error message display */}
        {error && (
          <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg animate-fade-in" id="error-display">
            <svg className="w-5 h-5 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm text-red-700">{error}</span>
          </div>
        )}

        {/* Submit button with loading spinner */}
        <button
          type="submit"
          id="predict-button"
          disabled={isLoading || message.trim().length < 3}
          className="btn-primary w-full flex items-center justify-center space-x-2 py-3 text-base"
        >
          {isLoading ? (
            <>
              <span className="spinner"></span>
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span>Classify Intent</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};

export default PredictForm;
