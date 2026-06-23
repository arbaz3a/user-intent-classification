/**
 * HistoryTable.jsx — Prediction history table
 *
 * Displays all past predictions in a clean table with columns:
 * Message (preview), Predicted Intent, Confidence, Date/Time, Delete button.
 * Shows an empty state message when no predictions exist.
 */

import { useState } from 'react';
import { deletePrediction } from '../services/api';

const HistoryTable = ({ predictions, onDelete }) => {
  const [deletingId, setDeletingId] = useState(null);

  // Handle delete with loading state
  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this prediction?')) {
      return;
    }

    try {
      setDeletingId(id);
      await deletePrediction(id);
      console.log('[HistoryTable] Deleted prediction:', id);
      // Notify parent to remove from state
      if (onDelete) onDelete(id);
    } catch (error) {
      console.error('[HistoryTable] Error deleting:', error.message);
      alert('Failed to delete prediction. Please try again.');
    } finally {
      setDeletingId(null);
    }
  };

  // Format date to readable string
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Truncate message for preview
  const truncateMessage = (msg, maxLen = 60) => {
    if (msg.length <= maxLen) return msg;
    return msg.substring(0, maxLen) + '...';
  };

  // Format intent for display
  const formatIntent = (intent) => {
    return intent
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Get confidence color
  const getConfidenceBadge = (confidence) => {
    const pct = Math.round(confidence * 100);
    if (pct >= 80) return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    if (pct >= 60) return 'bg-amber-50 text-amber-700 border-amber-200';
    return 'bg-red-50 text-red-700 border-red-200';
  };

  // ====== Empty state ======
  if (!predictions || predictions.length === 0) {
    return (
      <div className="card p-12 text-center" id="empty-state">
        <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-gray-600 mb-1">No Predictions Yet</h3>
        <p className="text-sm text-gray-400">
          Go to the Home page and classify some customer messages to see them here.
        </p>
      </div>
    );
  }

  // ====== Table ======
  return (
    <div className="card overflow-hidden" id="history-table">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="text-left px-6 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Message
              </th>
              <th className="text-left px-6 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Predicted Intent
              </th>
              <th className="text-center px-6 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Confidence
              </th>
              <th className="text-left px-6 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Date & Time
              </th>
              <th className="text-center px-6 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {predictions.map((prediction, index) => (
              <tr
                key={prediction._id}
                className="hover:bg-gray-50 transition-colors duration-150"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                {/* Message preview */}
                <td className="px-6 py-4 max-w-xs">
                  <p className="text-sm text-gray-900 font-medium" title={prediction.message}>
                    {truncateMessage(prediction.message)}
                  </p>
                </td>

                {/* Predicted Intent */}
                <td className="px-6 py-4">
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-primary-50 text-primary-700 text-sm font-medium border border-primary-200">
                    {formatIntent(prediction.predicted_intent)}
                  </span>
                </td>

                {/* Confidence */}
                <td className="px-6 py-4 text-center">
                  <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-sm font-semibold border ${getConfidenceBadge(prediction.confidence)}`}>
                    {Math.round(prediction.confidence * 100)}%
                  </span>
                </td>

                {/* Date/Time */}
                <td className="px-6 py-4">
                  <span className="text-sm text-gray-500">
                    {formatDate(prediction.timestamp)}
                  </span>
                </td>

                {/* Delete button */}
                <td className="px-6 py-4 text-center">
                  <button
                    onClick={() => handleDelete(prediction._id)}
                    disabled={deletingId === prediction._id}
                    className="btn-danger"
                    title="Delete prediction"
                  >
                    {deletingId === prediction._id ? (
                      <span className="flex items-center space-x-1">
                        <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <span>...</span>
                      </span>
                    ) : (
                      <span className="flex items-center space-x-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                        <span>Delete</span>
                      </span>
                    )}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Table footer with count */}
      <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          Showing {predictions.length} prediction{predictions.length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  );
};

export default HistoryTable;
