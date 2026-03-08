const fs = require('fs');
const content = fs.readFileSync('dashboard.json', 'utf8');
try {
  JSON.parse(content);
  console.log("DASHBOARD PARSE: OK");
} catch(e) {
  console.log("DASHBOARD PARSE ERROR:", e.message);
}
