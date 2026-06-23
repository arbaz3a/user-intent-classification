/**
 * db.js — MongoDB Atlas connection module
 *
 * Connects to MongoDB Atlas using the MONGO_URI from environment variables.
 * Logs success or failure message to the console.
 */

const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGO_URI);
    console.log(`[db.js] ✅ MongoDB Atlas connected: ${conn.connection.host}`);
  } catch (error) {
    console.error(`[db.js] ❌ MongoDB connection failed: ${error.message}`);
    process.exit(1); // Exit with failure code
  }
};

module.exports = connectDB;
