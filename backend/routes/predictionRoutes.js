/**
 * predictionRoutes.js — Express routes for prediction endpoints
 *
 * Routes:
 *   POST   /api/predictions       — Create a new prediction
 *   GET    /api/predictions        — Get all predictions
 *   GET    /api/predictions/stats  — Get prediction statistics
 *   DELETE /api/predictions/:id    — Delete a prediction by ID
 *
 * NOTE: /stats must come BEFORE /:id to prevent Express from
 * interpreting "stats" as a MongoDB ObjectId.
 */

const express = require('express');
const router = express.Router();

const {
  createPrediction,
  getAllPredictions,
  deletePrediction,
  getStats
} = require('../controllers/predictionController');

// GET /api/predictions/stats — must be defined BEFORE /:id route
router.get('/stats', getStats);

// POST /api/predictions — create a new prediction
router.post('/', createPrediction);

// GET /api/predictions — get all predictions
router.get('/', getAllPredictions);

// DELETE /api/predictions/:id — delete a prediction by ID
router.delete('/:id', deletePrediction);

module.exports = router;
