/**
 * History.jsx — Prediction history page
 *
 * Shows aggregate stats at the top (Total Predictions, Most Common Intent,
 * Average Confidence) as 3 cards, followed by the full history table.
 * Includes a refresh button to reload data.
 */

import { useState, useEffect } from 'react';
import HistoryTable from '../components/HistoryTable';
import { getAllPredictions, getStats } from '../services/api';

const History = () => {
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  // Fetch all data on mount
  useEffect(() => {
    fetchData();
  }, []);

  // Fetch predictions and stats
  const fetchData = async () => {
    setIsLoading(true);
    setError('');

    try {
      console.log('[History] Fetching data...');

      // Fetch both in parallel
      const [predictionsRes, statsRes] = await Promise.all([
        getAllPredictions(),
        getStats()
      ]);

      setPredictions(predictionsRes.data || []);
      setStats(statsRes.data || null);

      console.log('[History] Data loaded:', {
        predictions: predictionsRes.count,
        stats: statsRes.data
      });
    } catch (err) {
      console.error('[History] Error fetching data:', err.message);
      setError('Failed to load prediction history. Make sure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle prediction deletion — remove from local state
  const handleDelete = (deletedId) => {
    setPredictions(prev => prev.filter(p => p._id !== deletedId));
    // Refresh stats after deletion
    getStats().then(res => setStats(res.data)).catch(console.error);
  };

  // Format intent for display
  const formatIntent = (intent) => {
    if (!intent || intent === 'N/A') return 'N/A';
    return intent
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="min-h-[calc(100vh-64px)] bg-gray-50">
      {/* Page Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900" id="history-title">
                Prediction History
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                View all past predictions, statistics, and manage your history.
              </p>
            </div>

            {/* Refresh button */}
            <button
              onClick={fetchData}
              disabled={isLoading}
              id="refresh-button"
              className="btn-primary flex items-center space-x-2 self-start"
            >
              <svg
                className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>{isLoading ? 'Loading...' : 'Refresh'}</span>
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-3 animate-fade-in">
            <svg className="w-5 h-5 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm text-red-700">{error}</span>
          </div>
        )}

        {/* Stats Cards — 3 cards in a row */}
        {stats && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 mb-8 animate-fade-in" id="stats-section">
            {/* Total Predictions */}
            <div className="card p-6">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-primary-50 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Predictions</p>
                  <p className="text-2xl font-bold text-gray-900" id="stat-total">
                    {stats.totalPredictions}
                  </p>
                </div>
              </div>
            </div>

            {/* Most Common Intent */}
            <div className="card p-6">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-emerald-50 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Most Common Intent</p>
                  <p className="text-lg font-bold text-gray-900 truncate" id="stat-common-intent" title={stats.mostCommonIntent}>
                    {formatIntent(stats.mostCommonIntent)}
                  </p>
                </div>
              </div>
            </div>

            {/* Average Confidence */}
            <div className="card p-6">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-amber-50 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Average Confidence</p>
                  <p className="text-2xl font-bold text-gray-900" id="stat-avg-confidence">
                    {stats.averageConfidence
                      ? `${Math.round(stats.averageConfidence * 100)}%`
                      : 'N/A'
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {isLoading && !predictions.length && (
          <div className="space-y-4 animate-pulse">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 mb-8">
              {[1, 2, 3].map(i => (
                <div key={i} className="card p-6">
                  <div className="h-4 bg-gray-200 rounded w-1/3 mb-2"></div>
                  <div className="h-8 bg-gray-200 rounded w-1/2"></div>
                </div>
              ))}
            </div>
            <div className="card p-6">
              <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="h-12 bg-gray-100 rounded mb-2"></div>
              ))}
            </div>
          </div>
        )}

        {/* History Table */}
        {!isLoading && (
          <div className="animate-fade-in">
            <HistoryTable predictions={predictions} onDelete={handleDelete} />
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
