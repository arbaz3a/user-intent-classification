/**
 * App.jsx — Root application component
 *
 * Sets up React Router with two routes:
 * - / (Home) — Main prediction form and result display
 * - /history — Prediction history and statistics
 *
 * Navbar is always visible at the top of every page.
 */

import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import History from './pages/History';

const App = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navbar — always visible */}
      <Navbar />

      {/* Page Routes */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/history" element={<History />} />
      </Routes>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="text-xs text-gray-400">
              © {new Date().getFullYear()} Intent Classifier — ML-Powered Customer Support Intent Classification
            </p>
            <div className="flex items-center space-x-1.5">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-gray-400">System Online</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
