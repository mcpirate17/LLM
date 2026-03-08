const fs = require('fs');

const fixFiles = [
  'research/dashboard/src/components/report/NegativeResultsSummary.js',
  'research/dashboard/src/components/LiveFeed.js',
  'research/dashboard/src/components/LearningPanel.js',
  'research/dashboard/src/components/StrategyAdvisor.js',
  'research/dashboard/src/components/ExperimentList.js',
  'research/dashboard/src/hooks/useReportGallery.js',
  'research/dashboard/src/hooks/useAriaData.js',
];

for (const file of fixFiles) {
  let code = fs.readFileSync(file, 'utf8');
  if (file.includes('ExperimentList.js')) {
     code = code.replace(/fetch\(\`/g, 'apiCall(`');
     code = code.replace(/fetch\(\'/g, "apiCall('");
  } else {
     code = code.replace(/fetch\(\`\$\{API_BASE\}/g, 'apiCall(`');
     code = code.replace(/fetch\(\`\$\{apiBase\}/g, 'apiCall(`');
     code = code.replace(/fetch\(\`\$\{base\}/g, 'apiCall(`');
  }
  if (!code.includes('apiCall')) {
      code = code.replace(/fetch\(/g, "apiCall(");
  }
  if (!code.includes('import ') || (!code.includes('apiCall') && !file.includes('apiService'))) {
      if (!code.includes('import { apiCall }')) {
          code = `import { apiCall } from "../services/apiService";\n` + code;
      }
  }
  
  // Custom fixer for the import path differences
  if (file.includes('hooks/')) {
      code = code.replace(`from "../services/apiService"`, `from "../services/apiService"`);
  }
  fs.writeFileSync(file, code);
}
