/**
 * api.js — API service layer
 *
 * Centralizes all HTTP calls to the Express backend.
 * Base URL points to localhost:5000 (proxied in dev via Vite config).
 */

import axios from 'axios';

// Base URL for the Express backend API
const API_BASE_URL = 'http://localhost:5000';

// Create an axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * predictIntent — POST /api/predictions
 *
 * Sends a customer message to the backend, which forwards it to the
 * FastAPI ML service for prediction, then saves and returns the result.
 *
 * @param {string} message - The customer support message
 * @returns {Promise<Object>} - The prediction object
 */
export const predictIntent = async (message) => {
  try {
    console.log('[api] Sending prediction request:', message);
    const response = await api.post('/api/predictions', { message });
    console.log('[api] Prediction response:', response.data);
    return response.data;
  } catch (error) {
    console.error('[api] Error predicting intent:', error.response?.data || error.message);
    throw error;
  }
};

/**
 * getAllPredictions — GET /api/predictions
 *
 * Retrieves all past predictions sorted by timestamp descending.
 *
 * @returns {Promise<Object>} - Array of prediction objects
 */
export const getAllPredictions = async () => {
  try {
    console.log('[api] Fetching all predictions...');
    const response = await api.get('/api/predictions');
    console.log('[api] Predictions fetched:', response.data.count);
    return response.data;
  } catch (error) {
    console.error('[api] Error fetching predictions:', error.response?.data || error.message);
    throw error;
  }
};

/**
 * deletePrediction — DELETE /api/predictions/:id
 *
 * Deletes a single prediction by its MongoDB _id.
 *
 * @param {string} id - The prediction MongoDB _id
 * @returns {Promise<Object>} - Success response
 */
export const deletePrediction = async (id) => {
  try {
    console.log('[api] Deleting prediction:', id);
    const response = await api.delete(`/api/predictions/${id}`);
    console.log('[api] Prediction deleted:', id);
    return response.data;
  } catch (error) {
    console.error('[api] Error deleting prediction:', error.response?.data || error.message);
    throw error;
  }
};

/**
 * getStats — GET /api/predictions/stats
 *
 * Retrieves aggregate statistics: total predictions, most common intent,
 * and average confidence.
 *
 * @returns {Promise<Object>} - Stats object
 */
export const getStats = async () => {
  try {
    console.log('[api] Fetching stats...');
    const response = await api.get('/api/predictions/stats');
    console.log('[api] Stats fetched:', response.data);
    return response.data;
  } catch (error) {
    console.error('[api] Error fetching stats:', error.response?.data || error.message);
    throw error;
  }
};
