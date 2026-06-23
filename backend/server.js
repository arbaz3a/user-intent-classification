/**
 * server.js — Express application entry point
 *
 * Sets up the Express server with:
 * - CORS middleware for cross-origin requests
 * - JSON body parsing
 * - MongoDB Atlas connection
 * - Prediction routes
 * - Global error handling
 *
 * Runs on port 5000 (configurable via .env).
 */

const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./config/db');
const predictionRoutes = require('./routes/predictionRoutes');
const errorHandler = require('./middleware/errorHandler');

// Load environment variables from .env file
dotenv.config();

// Connect to MongoDB Atlas
connectDB();

// Initialize Express app
const app = express();

// ============================================================
// Middleware
// ============================================================

// Enable CORS for all origins (frontend runs on a different port)
app.use(cors());

// Parse incoming JSON request bodies
app.use(express.json());

// Log all incoming requests for debugging
app.use((req, res, next) => {
  console.log(`[server] ${req.method} ${req.originalUrl}`);
  next();
});

// ============================================================
// Routes
// ============================================================

// Mount prediction routes at /api/predictions
app.use('/api/predictions', predictionRoutes);

// Basic health check route
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'backend' });
});

// ============================================================
// Error handling (must be AFTER routes)
// ============================================================

app.use(errorHandler);

// ============================================================
// Start server
// ============================================================

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`[server] 🚀 Backend server running on port ${PORT}`);
  console.log(`[server] 📡 API available at http://localhost:${PORT}/api/predictions`);
  console.log(`[server] 🤖 FastAPI ML service expected at ${process.env.FASTAPI_URL || 'http://localhost:8000'}`);
});
