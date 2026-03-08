import React from 'react';
import { CheckCircle2, Loader, Circle, XCircle } from 'lucide-react';

const STAGE_ORDER = ['conversion', 'profiling', 'compilation', 'sandbox', 'compression', 'fingerprint', 'novelty'];

const STAGE_LABELS = {
  conversion: 'Conversion',
  profiling: 'Profiling',
  compilation: 'Compilation',
  sandbox: 'Sandbox Eval',
  compression: 'Compression',
  fingerprint: 'Fingerprint',
  novelty: 'Novelty',
};

function StageIcon({ status }) {
  if (status === 'done') return <CheckCircle2 size={14} color="#24d1a0" />;
  if (status === 'running') return <Loader size={14} className="animate-spin" />;
  if (status === 'error') return <XCircle size={14} color="#ff5050" />;
  return <Circle size={14} />;
}

const EvalStepper = ({ stageMap }) => {
  return (
    <div className="eval-stepper">
      {STAGE_ORDER.map((name) => {
        const s = stageMap[name];
        const st = s?.status || 'pending';
        return (
          <div key={name} className={`eval-step stage-${st}`}>
            <span className="step-icon"><StageIcon status={st} /></span>
            <span className="step-name">{STAGE_LABELS[name]}</span>
            {s?.elapsed_ms != null && <span className="step-time">{s.elapsed_ms.toFixed(0)}ms</span>}
          </div>
        );
      })}
    </div>
  );
};

export default EvalStepper;
