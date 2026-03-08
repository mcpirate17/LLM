// Script to patch App.js
const fs = require('fs');
const file = 'research/dashboard/src/App.js';
let code = fs.readFileSync(file, 'utf8');

// The main issue is DRY violations: App.js doesn't use apiCall for fetches!
// Let me replace all fetch(`\${API_BASE} with apiCall(
code = code.replace(/fetch\(`\$\{API_BASE\}\/api\/fingerprint\/resolve\?value=\$\{encodeURIComponent\(value\)\}`\)/g, "apiCall(`/api/fingerprint/resolve?value=\${encodeURIComponent(value)}\`)");
code = code.replace(/fetch\(`\$\{API_BASE\}\//g, "apiCall(`/");

// also add import apiCall if not present
if (!code.includes('apiCall')) {
  code = "import { apiCall } from './services/apiService';\n" + code;
}

fs.writeFileSync(file, code);
