/**
 * errorHandler.js — Global error handling middleware
 *
 * Catches all unhandled errors and returns a consistent JSON response
 * with the error message and appropriate HTTP status code.
 */

const errorHandler = (err, req, res, next) => {
  console.error('[errorHandler] Error caught:', err.message);
  console.error('[errorHandler] Stack:', err.stack);

  // Default to 500 if no status code was set
  const statusCode = res.statusCode === 200 ? 500 : res.statusCode;

  res.status(statusCode).json({
    success: false,
    error: err.message || 'Internal Server Error',
    // Include stack trace only in development
    stack: process.env.NODE_ENV === 'production' ? undefined : err.stack
  });
};

module.exports = errorHandler;
