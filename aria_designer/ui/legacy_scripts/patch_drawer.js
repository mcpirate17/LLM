// Script to patch ArchitectureDrawer.js
const fs = require('fs');
const file = 'research/dashboard/src/components/ArchitectureDrawer.js';
let code = fs.readFileSync(file, 'utf8');

// Replace the two useEffects with a synchronized generic one
const searchStr = `  // Ensure designer services are running before loading iframe.
  useEffect(() => {
    let cancelled = false;
    const ensure = async () => {
      setStartingDesigner(true);
      setBooting(true);
      setBridgeStep('starting-services');
      setError(null);
      try {
        const res = await apiCall('/api/designer/ensure-running', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ force_restart: false }),
        });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok || payload?.ok === false) {
          throw new Error(payload?.error || \`HTTP \${res.status}\`);
        }
      } catch (err) {
        if (!cancelled) {
          setError(\`Failed to start Aria Designer services: \${err.message}\`);
        }
      } finally {
        if (!cancelled) setStartingDesigner(false);
      }
    };
    ensure();
    return () => { cancelled = true; };
  }, []);

  // Initial load
  useEffect(() => {
    if (startingDesigner) return;
    // Blank-canvas mode: open designer directly without fetching a source result.
    if (!resultId) {
      setGraphInfo(null);
      setSourceGraphCheck(null);
      setLoading(false);
      setBooting(false);
      setBridgeStep('ready');
      return;
    }
    setLoading(true);
    setBooting(true);
    setBridgeStep('fetching-source');
    apiCall(\`/api/programs/\${resultId}\`)
      .then((r) => r.json())
      .then((data) => {
        setGraphInfo(data);
        setSourceGraphCheck(analyzeResearchGraph(data.graph_json_parsed));
        setBridgeStep('loading-iframe');
        setBooting(false);
      })
      .catch((err) => {
        setError(\`Failed to fetch source graph: \${err.message}\`);
        setLoading(false);
        setBooting(false);
      });
  }, [resultId, startingDesigner]);`;

const replacement = `  // Ensure designer services and fetch source graph in parallel.
  useEffect(() => {
    let cancelled = false;
    
    setBooting(true);
    setError(null);
    setBridgeStep('starting-services');    
    setLoading(true);

    const checkDesigner = apiCall('/api/designer/ensure-running', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force_restart: false }),
    }).then(res => res.json().then(payload => {
      if (!res.ok || payload?.ok === false) {
        throw new Error(payload?.error || \`HTTP \${res.status}\`);
      }
    }));

    const fetchSource = resultId 
      ? apiCall(\`/api/programs/\${resultId}\`).then(r => r.json())
      : Promise.resolve(null);

    Promise.all([checkDesigner, fetchSource])
      .then(([_, sourceData]) => {
        if (cancelled) return;
        setStartingDesigner(false);
        if (sourceData) {
          setGraphInfo(sourceData);
          setSourceGraphCheck(analyzeResearchGraph(sourceData.graph_json_parsed));
        } else {
          setGraphInfo(null);
          setSourceGraphCheck(null);
        }
        setBridgeStep(sourceData ? 'loading-iframe' : 'ready');
        setBooting(false);
        if (!sourceData) setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setStartingDesigner(false);
        setError(\`Failed to initialize: \${err.message}\`);
        setLoading(false);
        setBooting(false);
      });

    return () => { cancelled = true; };
  }, [resultId]);`;

code = code.replace(searchStr, replacement);
fs.writeFileSync(file, code);
