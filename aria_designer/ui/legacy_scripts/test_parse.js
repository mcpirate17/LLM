const fs = require('fs');
const data = fs.readFileSync('dashboard.json', 'utf8');
try {
  JSON.parse(data);
  console.log("SUCCESS");
} catch(e) {
  console.log("FAIL", e.message);
}
