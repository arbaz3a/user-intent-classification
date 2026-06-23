/**
 * Prediction.js — Mongoose schema/model for prediction records
 *
 * Stores each prediction made by the ML model, including the original
 * message, predicted intent, confidence score, timestamp, and processing time.
 */

const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
  // The original customer support message
  message: {
    type: String,
    required: [true, 'Message is required'],
    trim: true
  },

  // The predicted intent label (e.g., "cancel_order")
  predicted_intent: {
    type: String,
    required: [true, 'Predicted intent is required']
  },

  // Model confidence score (0 to 1)
  confidence: {
    type: Number
  },

  // When the prediction was made
  timestamp: {
    type: Date,
    default: Date.now
  },

  // How long the ML model took to process (in milliseconds)
  processing_time_ms: {
    type: Number
  }
});

module.exports = mongoose.model('Prediction', PredictionSchema);
