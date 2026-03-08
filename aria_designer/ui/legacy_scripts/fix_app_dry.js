const fs = require('fs');
let code = fs.readFileSync('aria_designer/ui/src/App.jsx', 'utf8');

const compileDry = `      let usedDesignerApi = true
      try {
        const res = await apiCall(\`/api/v1/workflows/compile\`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow, target: 'cpu' }),
        })
        if (!res.ok) throw new Error(\`compile \${res.status}\`)
        data = await res.json()
      } catch {
        usedDesignerApi = false
        const res = await apiCall(\`/api/v1/workflows/compile\`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow, target: 'cpu' }),
        })
        data = await res.json()
      }`;

const compileFixed = `      let usedDesignerApi = true
      const res = await apiCall(\`/api/v1/workflows/compile\`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, target: 'cpu' }),
      })
      if (!res.ok) throw new Error(\`compile \${res.status}\`)
      data = await res.json()`;

const validateDry = `      try {
        const res = await apiCall(\`/api/v1/workflows/validate\`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        if (!res.ok) throw new Error(\`validate \${res.status}\`)
        data = await res.json()
      } catch {
        const res = await apiCall(\`/api/v1/workflows/validate\`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        data = await res.json()
      }`;

const validateFixed = `      const res = await apiCall(\`/api/v1/workflows/validate\`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
      })
      if (!res.ok) {
        // Only throw if validation hard-failed to return details
        if (res.status !== 400 && res.status !== 422) {
           throw new Error(\`validate \${res.status}\`)
        }
      }
      data = await res.json()`;

code = code.replace(compileDry, compileFixed);
code = code.replace(validateDry, validateFixed);
fs.writeFileSync('aria_designer/ui/src/App.jsx', code);
