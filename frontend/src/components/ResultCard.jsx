/**
 * ResultCard.jsx — Prediction result display card
 *
 * Shows the predicted intent in large bold text, confidence as a
 * percentage with a colored progress bar (green > 80%, yellow > 60%, red < 60%),
 * and processing time in milliseconds. Smooth fade-in animation on appear.
 */

const ResultCard = ({ prediction }) => {
  if (!prediction) return null;

  const { predicted_intent, confidence, processing_time_ms } = prediction;

  // Convert confidence to percentage
  const confidencePercent = Math.round(confidence * 100);

  // Determine color based on confidence level
  const getConfidenceColor = (pct) => {
    if (pct >= 80) return { bg: 'bg-emerald-500', text: 'text-emerald-700', light: 'bg-emerald-50', border: 'border-emerald-200' };
    if (pct >= 60) return { bg: 'bg-amber-500', text: 'text-amber-700', light: 'bg-amber-50', border: 'border-amber-200' };
    return { bg: 'bg-red-500', text: 'text-red-700', light: 'bg-red-50', border: 'border-red-200' };
  };

  const colors = getConfidenceColor(confidencePercent);

  // Format the intent string for display (replace underscores with spaces, title case)
  const formatIntent = (intent) => {
    return intent
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="card p-6 animate-fade-in" id="result-card">
      {/* Header */}
      <div className="flex items-center space-x-2 mb-4">
        <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
          <svg className="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
          Prediction Result
        </h3>
      </div>

      {/* Predicted Intent — large bold text */}
      <div className="mb-6">
        <p className="text-sm text-gray-500 mb-1">Predicted Intent</p>
        <p className="text-2xl font-bold text-gray-900" id="predicted-intent">
          {formatIntent(predicted_intent)}
        </p>
        <p className="text-xs text-gray-400 mt-1 font-mono">
          {predicted_intent}
        </p>
      </div>

      {/* Confidence bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm text-gray-500">Confidence</p>
          <span className={`text-sm font-bold ${colors.text}`} id="confidence-value">
            {confidencePercent}%
          </span>
        </div>

        {/* Progress bar container */}
        <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full rounded-full progress-bar-fill ${colors.bg}`}
            style={{ width: `${confidencePercent}%` }}
            role="progressbar"
            aria-valuenow={confidencePercent}
            aria-valuemin={0}
            aria-valuemax={100}
          ></div>
        </div>

        {/* Confidence label */}
        <p className={`mt-2 text-xs font-medium ${colors.text} ${colors.light} ${colors.border} border inline-block px-2.5 py-1 rounded-full`}>
          {confidencePercent >= 80 ? '● High Confidence' : confidencePercent >= 60 ? '● Medium Confidence' : '● Low Confidence'}
        </p>
      </div>

      {/* Processing time */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-100">
        <div className="flex items-center space-x-2">
          <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm text-gray-500">Processing Time</span>
        </div>
        <span className="text-sm font-semibold text-gray-700" id="processing-time">
          {processing_time_ms?.toFixed(1)} ms
        </span>
      </div>
    </div>
  );
};

export default ResultCard;
