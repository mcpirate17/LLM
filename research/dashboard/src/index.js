import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const CHUNK_RELOAD_KEY = 'aria-dashboard-chunk-reload-attempted';
const CHUNK_RELOAD_TTL_MS = 60000;

function isChunkLoadError(error) {
  const message = String(error?.message || '');
  const name = String(error?.name || '');
  return (
    name === 'ChunkLoadError' ||
    /Loading chunk \d+ failed/i.test(message) ||
    /ChunkLoadError/i.test(message)
  );
}

class RootErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error("Root ErrorBoundary caught an error", error, errorInfo);
    const lastReload = Number(window.sessionStorage.getItem(CHUNK_RELOAD_KEY) || 0);
    if (isChunkLoadError(error) && Date.now() - lastReload > CHUNK_RELOAD_TTL_MS) {
      window.sessionStorage.setItem(CHUNK_RELOAD_KEY, String(Date.now()));
      window.location.reload();
    }
  }
  componentDidMount() {
    const lastReload = Number(window.sessionStorage.getItem(CHUNK_RELOAD_KEY) || 0);
    if (lastReload && Date.now() - lastReload > CHUNK_RELOAD_TTL_MS) {
      window.sessionStorage.removeItem(CHUNK_RELOAD_KEY);
    }
  }
  render() {
    if (this.state.hasError) {
      const isChunkError = isChunkLoadError(this.state.error);
      return (
        <div style={{
          padding: 40, maxWidth: 600, margin: '60px auto',
          background: '#161b22', borderRadius: 8,
          border: '1px solid #f85149', color: '#e6edf3',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        }}>
          <h2 style={{ color: '#f85149', marginTop: 0 }}>Dashboard crashed</h2>
          <p style={{ color: '#8b949e', fontSize: 14, lineHeight: 1.6 }}>
            An unexpected error caused the dashboard to stop rendering.
          </p>
          <pre style={{
            background: '#0d1117', padding: 12, borderRadius: 6,
            fontSize: 12, color: '#f85149', overflow: 'auto', maxHeight: 200,
          }}>
            {this.state.error?.message}
          </pre>
          <button
            onClick={() => {
              if (isChunkError) {
                window.location.reload();
                return;
              }
              this.setState({ hasError: false, error: null });
            }}
            style={{
              marginTop: 12, padding: '8px 16px', borderRadius: 6,
              background: '#238636', color: '#fff', border: 'none',
              cursor: 'pointer', fontSize: 13, fontWeight: 600,
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <RootErrorBoundary>
    <App />
  </RootErrorBoundary>
);
