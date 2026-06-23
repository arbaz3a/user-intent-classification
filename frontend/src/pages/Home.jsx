/**
 * Home.jsx — Main page with hero section, prediction form, and result card
 *
 * Flow:
 * 1. User types a customer support message in the textarea
 * 2. Clicks "Classify Intent" button
 * 3. Request goes to Express backend → FastAPI ML service → back
 * 4. Result card appears with fade-in animation showing the prediction
 */

import { useState } from 'react';
import PredictForm from '../components/PredictForm';
import ResultCard from '../components/ResultCard';
import { predictIntent } from '../services/api';

const Home = () => {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Handle prediction request
  const handlePredict = async (message) => {
    setIsLoading(true);
    setPrediction(null); // Clear previous result

    try {
      console.log('[Home] Predicting intent for:', message);
      const response = await predictIntent(message);

      if (response.success) {
        setPrediction(response.data);
        console.log('[Home] Prediction received:', response.data);
      }
    } catch (error) {
      console.error('[Home] Prediction error:', error);
      throw error; // Re-throw so PredictForm can display the error
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-64px)] bg-gray-50">
      {/* Hero Section */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
          {/* Badge */}
          <div className="inline-flex items-center px-3 py-1 rounded-full bg-primary-50 border border-primary-200 mb-6">
            <div className="w-2 h-2 bg-primary-500 rounded-full mr-2 animate-pulse"></div>
            <span className="text-xs font-semibold text-primary-700 uppercase tracking-wide">
              ML-Powered Classification
            </span>
          </div>

          {/* Title */}
          <h1 className="text-4xl sm:text-5xl font-extrabold text-gray-900 tracking-tight mb-4" id="hero-title">
            User Intent
            <span className="text-primary-600"> Classification</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg text-gray-500 max-w-2xl mx-auto leading-relaxed">
            Type a customer support message and our machine learning model will instantly
            predict the user's intent with confidence scoring.
          </p>

          {/* Feature pills */}
          <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
            <div className="flex items-center space-x-1.5 px-3 py-1.5 bg-gray-50 rounded-full border border-gray-200">
              <svg className="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="text-xs font-medium text-gray-600">TF-IDF + SVM Model</span>
            </div>
            <div className="flex items-center space-x-1.5 px-3 py-1.5 bg-gray-50 rounded-full border border-gray-200">
              <svg className="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="text-xs font-medium text-gray-600">Real-time Prediction</span>
            </div>
            <div className="flex items-center space-x-1.5 px-3 py-1.5 bg-gray-50 rounded-full border border-gray-200">
              <svg className="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="text-xs font-medium text-gray-600">Confidence Scoring</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Prediction Form */}
        <div className="card p-6 mb-8">
          <PredictForm onPredict={handlePredict} isLoading={isLoading} />
        </div>

        {/* Result Card — appears with fade-in animation after prediction */}
        {prediction && (
          <div className="animate-slide-up">
            <ResultCard prediction={prediction} />
          </div>
        )}

        {/* Example messages hint */}
        {!prediction && !isLoading && (
          <div className="text-center mt-6 animate-fade-in">
            <p className="text-sm text-gray-400 mb-3">Try one of these example messages:</p>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                'I want to cancel my order',
                'Where is my refund?',
                'How do I change my shipping address?',
                'My product arrived damaged'
              ].map((example, i) => (
                <button
                  key={i}
                  onClick={() => {
                    document.getElementById('message-input').value = example;
                    // Trigger React's onChange by dispatching an input event
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                      window.HTMLTextAreaElement.prototype, 'value'
                    ).set;
                    const input = document.getElementById('message-input');
                    nativeInputValueSetter.call(input, example);
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                  }}
                  className="text-xs px-3 py-1.5 bg-white border border-gray-200 rounded-full text-gray-500 hover:text-primary-600 hover:border-primary-300 transition-all duration-200"
                >
                  "{example}"
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;
