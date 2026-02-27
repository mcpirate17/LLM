/**
 * Shared configuration constants for the Aria Designer UI.
 * Import from here instead of redefining in individual components.
 */

const getDesignerApiBase = () => {
  if (import.meta.env.VITE_DESIGNER_API_BASE) return import.meta.env.VITE_DESIGNER_API_BASE;
  if (typeof window === 'undefined') return 'http://127.0.0.1:8091';
  const url = new URL(window.location.href);
  // When served from the dashboard proxy (/designer-proxy/ on port 5000),
  // route API calls through the same origin so the proxy can forward them.
  // We check for port 5000 specifically as the dashboard port.
  if (url.pathname.startsWith('/designer-proxy') || url.port === '5000') {
    return ''; // Use relative paths to hit dashboard proxy
  }
  return 'http://127.0.0.1:8091';
};

export const DESIGNER_API_BASE = getDesignerApiBase();
