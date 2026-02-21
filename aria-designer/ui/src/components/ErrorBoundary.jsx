import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '20px', color: '#ff5050', background: '#101b2b', borderRadius: '8px', border: '1px solid #1f3147' }}>
          <h3>Something went wrong in the {this.props.name || 'component'}.</h3>
          <p style={{ fontSize: '12px', opacity: 0.8 }}>{this.state.error?.toString()}</p>
          <button 
            onClick={() => this.setState({ hasError: false })}
            style={{ padding: '4px 8px', background: '#17a3ff', border: 'none', borderRadius: '4px', color: 'white', cursor: 'pointer' }}
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
