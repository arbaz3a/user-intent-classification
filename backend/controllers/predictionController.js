/**
 * predictionController.js — Business logic for prediction CRUD operations
 *
 * Handles creating predictions (by calling the FastAPI ML service),
 * retrieving all predictions, deleting predictions, and computing stats.
 */

const axios = require('axios');
const Prediction = require('../models/Prediction');

// FastAPI ML service URL from environment
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

/**
 * createPrediction — POST /api/predictions
 *
 * 1. Receives a message from the request body
 * 2. Sends it to the FastAPI ML service at /predict
 * 3. Gets back the predicted intent and confidence
 * 4. Saves the full prediction record to MongoDB
 * 5. Returns the saved prediction object
 */
const createPrediction = async (req, res, next) => {
  try {
    const { message } = req.body;

    // Validate input
    if (!message || message.trim().length < 3) {
      return res.status(400).json({
        success: false,
        error: 'Message is required and must be at least 3 characters long'
      });
    }

    console.log(`[controller] Creating prediction for: "${message}"`);

    // Call the FastAPI ML service
    const mlResponse = await axios.post(`${FASTAPI_URL}/predict`, {
      message: message.trim()
    });

    console.log(`[controller] ML API response:`, mlResponse.data);

    const { intent, confidence, processing_time_ms } = mlResponse.data;

    // Save to MongoDB
    const prediction = await Prediction.create({
      message: message.trim(),
      predicted_intent: intent,
      confidence: confidence,
      processing_time_ms: processing_time_ms
    });

    console.log(`[controller] Prediction saved to DB: ${prediction._id}`);

    res.status(201).json({
      success: true,
      data: prediction
    });
  } catch (error) {
    console.error(`[controller] Error creating prediction:`, error.message);

    // Check if it's an axios error (ML service unavailable)
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        success: false,
        error: 'ML service is unavailable. Make sure the FastAPI server is running on port 8000.'
      });
    }

    next(error);
  }
};

/**
 * getAllPredictions — GET /api/predictions
 *
 * Returns all predictions sorted by timestamp descending (newest first).
 */
const getAllPredictions = async (req, res, next) => {
  try {
    console.log('[controller] Fetching all predictions...');

    const predictions = await Prediction.find()
      .sort({ timestamp: -1 }); // Newest first

    console.log(`[controller] Found ${predictions.length} predictions`);

    res.status(200).json({
      success: true,
      count: predictions.length,
      data: predictions
    });
  } catch (error) {
    console.error('[controller] Error fetching predictions:', error.message);
    next(error);
  }
};

/**
 * deletePrediction — DELETE /api/predictions/:id
 *
 * Deletes a single prediction by its MongoDB _id.
 */
const deletePrediction = async (req, res, next) => {
  try {
    const { id } = req.params;

    console.log(`[controller] Deleting prediction: ${id}`);

    const prediction = await Prediction.findByIdAndDelete(id);

    if (!prediction) {
      return res.status(404).json({
        success: false,
        error: 'Prediction not found'
      });
    }

    console.log(`[controller] Prediction deleted: ${id}`);

    res.status(200).json({
      success: true,
      data: {}
    });
  } catch (error) {
    console.error('[controller] Error deleting prediction:', error.message);
    next(error);
  }
};

/**
 * getStats — GET /api/predictions/stats
 *
 * Returns aggregate statistics:
 * - Total number of predictions
 * - Most common intent
 * - Average confidence score
 */
const getStats = async (req, res, next) => {
  try {
    console.log('[controller] Computing prediction stats...');

    const totalPredictions = await Prediction.countDocuments();

    // Find the most common intent using MongoDB aggregation
    const intentAggregation = await Prediction.aggregate([
      {
        $group: {
          _id: '$predicted_intent',
          count: { $sum: 1 }
        }
      },
      { $sort: { count: -1 } },
      { $limit: 1 }
    ]);

    // Calculate average confidence
    const confidenceAggregation = await Prediction.aggregate([
      {
        $group: {
          _id: null,
          avgConfidence: { $avg: '$confidence' }
        }
      }
    ]);

    const mostCommonIntent = intentAggregation.length > 0
      ? intentAggregation[0]._id
      : 'N/A';

    const averageConfidence = confidenceAggregation.length > 0
      ? Math.round(confidenceAggregation[0].avgConfidence * 10000) / 10000
      : 0;

    const stats = {
      totalPredictions,
      mostCommonIntent,
      averageConfidence
    };

    console.log('[controller] Stats:', stats);

    res.status(200).json({
      success: true,
      data: stats
    });
  } catch (error) {
    console.error('[controller] Error computing stats:', error.message);
    next(error);
  }
};

module.exports = {
  createPrediction,
  getAllPredictions,
  deletePrediction,
  getStats
};
