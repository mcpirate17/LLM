import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

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
  }
  render() {
    if (this.state.hasError) {
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
            onClick={() => this.setState({ hasError: false, error: null })}
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
