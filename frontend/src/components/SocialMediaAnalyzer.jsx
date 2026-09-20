import { useState } from 'react';
import axios from 'axios';
import { Message, YouTube } from '@mui/icons-material';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

// The Audio tab was removed: the backend's transcribe_audio_wav2vec() could
// never run (its soundfile import and Wav2Vec2 processor were both commented
// out), so the tab advertised a feature that always returned an error.
const SocialMediaAnalyzer = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [text, setText] = useState('');
  const [link, setLink] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleTabChange = (index) => {
    setActiveTab(index);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async (inputType) => {
    if ((inputType === 'link' && !link) || (inputType === 'comment' && !text)) {
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      // Send real JSON. The previous version built a FormData body but set
      // Content-Type: application/json, so Flask's request.get_json() could
      // never parse it and every request came back 400.
      const payload = inputType === 'link' ? { link } : { text };
      const response = await axios.post(`${API_BASE}/analyze`, payload);
      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing:', err);
      setError(err?.response?.data?.error || 'Could not reach the analyzer.');
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (percentage) => {
    if (percentage <= 25) return 'bg-green-500';
    if (percentage <= 50) return 'bg-yellow-500';
    if (percentage <= 75) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const tabButton = (index, Icon, label) => (
    <button
      className={`px-6 py-2 mr-4 text-lg font-semibold rounded-lg ${
        activeTab === index ? 'bg-blue-500 text-white' : 'bg-gray-200'
      }`}
      onClick={() => handleTabChange(index)}
    >
      <Icon className="inline-block mr-2" /> {label}
    </button>
  );

  const spinner = (
    <div className="animate-spin border-4 border-t-4 border-white rounded-full w-5 h-5 mx-auto" />
  );

  return (
    <div className="max-w-4xl mx-auto py-8">
      <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-100 backdrop-blur-md">
        <h1 className="text-4xl font-bold text-center mb-4">Hate Speech Detection</h1>
        <p className="text-md text-center text-gray-600 mb-6">
          Analyze content for hateful speech in English, Hindi, Marathi and Bengali
        </p>

        <div className="flex justify-center mb-6">
          {tabButton(0, Message, 'Text')}
          {tabButton(1, YouTube, 'YouTube')}
        </div>

        {/* Text Analysis Tab */}
        {activeTab === 0 && (
          <div className="space-y-4">
            <textarea
              id="analyze-text"
              className="w-full p-3 border rounded-lg border-gray-300"
              rows={4}
              placeholder="Enter text to analyze (any of the four languages)"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <button
              className={`w-full py-2 bg-blue-500 text-white rounded-lg ${
                loading || !text ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              onClick={() => handleAnalyze('comment')}
              disabled={loading || !text}
            >
              {loading ? spinner : 'Analyze Text'}
            </button>
          </div>
        )}

        {/* YouTube Tab */}
        {activeTab === 1 && (
          <div className="space-y-4">
            <input
              id="analyze-link"
              type="text"
              className="w-full p-3 border rounded-lg border-gray-300"
              placeholder="Enter YouTube Video URL"
              value={link}
              onChange={(e) => setLink(e.target.value)}
            />
            <button
              className={`w-full py-2 bg-blue-500 text-white rounded-lg ${
                loading || !link ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              onClick={() => handleAnalyze('link')}
              disabled={loading || !link}
            >
              {loading ? spinner : 'Analyze Video Comments'}
            </button>
          </div>
        )}

        {/* Errors */}
        {error && (
          <div className="mt-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg text-red-800">
            {error}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="mt-8 space-y-4">
            <div className="p-4 border rounded-lg shadow-lg">
              <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>

              <div className="mb-4">
                <div className="flex justify-between mb-2">
                  <span>Hateful Content Detection</span>
                  <span className="tabular-nums">
                    {result.hateSpeechPercentage.toFixed(1)}%
                  </span>
                </div>
                <div className="relative pt-1">
                  <div className="w-full h-2 rounded-lg bg-gray-200 overflow-hidden">
                    <div
                      className={`h-2 rounded-lg ${getSeverityColor(
                        result.hateSpeechPercentage
                      )}`}
                      style={{ width: `${result.hateSpeechPercentage}%` }}
                    />
                  </div>
                </div>
                {result.commentsAnalyzed != null && (
                  <p className="text-sm text-gray-500 mt-2">
                    {result.commentsAnalyzed} item
                    {result.commentsAnalyzed === 1 ? '' : 's'} analyzed
                  </p>
                )}
              </div>

              {result.hateSpeechSamples?.length > 0 && (
                <div className="mb-4">
                  <strong className="text-red-700">Flagged as hateful</strong>
                  <ul className="list-disc pl-6 mt-1 space-y-1 text-sm">
                    {result.hateSpeechSamples.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.nonHateSpeechSamples?.length > 0 && (
                <div>
                  <strong className="text-green-700">Classified as clean</strong>
                  <ul className="list-disc pl-6 mt-1 space-y-1 text-sm">
                    {result.nonHateSpeechSamples.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SocialMediaAnalyzer;
