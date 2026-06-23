/**
 * Navbar.jsx — Top navigation bar
 *
 * Displays the app name "Intent Classifier" on the left and
 * navigation links (Home, History) on the right.
 * Dark background with clean white text styling.
 */

import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();

  // Helper to determine if a route is currently active
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-gray-900 shadow-lg sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Left side — Brand / Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="w-9 h-9 bg-primary-600 rounded-lg flex items-center justify-center shadow-md group-hover:bg-primary-500 transition-colors duration-200">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <span className="text-white font-bold text-lg tracking-tight">
              Intent Classifier
            </span>
          </Link>

          {/* Right side — Navigation links */}
          <div className="flex items-center space-x-1">
            <Link
              to="/"
              id="nav-home"
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive('/')
                  ? 'bg-primary-600 text-white shadow-md'
                  : 'text-gray-300 hover:text-white hover:bg-gray-800'
              }`}
            >
              <span className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
                <span>Home</span>
              </span>
            </Link>

            <Link
              to="/history"
              id="nav-history"
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive('/history')
                  ? 'bg-primary-600 text-white shadow-md'
                  : 'text-gray-300 hover:text-white hover:bg-gray-800'
              }`}
            >
              <span className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>History</span>
              </span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
