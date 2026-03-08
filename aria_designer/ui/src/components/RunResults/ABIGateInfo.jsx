import React from 'react';

const ABIGateInfo = ({ abiProbe }) => {
  if (!abiProbe) return null;

  const abiParityAttempted = Boolean(abiProbe.parity_attempted);
  const abiParityPass = abiProbe.parity_pass;
  const abiPrimaryUsed = Boolean(abiProbe.primary_used);
  const abiMode = abiProbe.mode || (abiPrimaryUsed ? 'primary_forward_only' : 'probe_only');
  const abiParityMaxAbs = Number(abiProbe.parity_max_abs_diff);
  const abiParityThreshold = Number(abiProbe.parity_max_abs_threshold);
  const abiSampleRate = Number(abiProbe.parity_sample_rate);
  
  const abiParityMaxAbsText = Number.isFinite(abiParityMaxAbs) ? abiParityMaxAbs.toExponential(2) : '-';
  const abiParityThresholdText = Number.isFinite(abiParityThreshold) ? abiParityThreshold.toExponential(2) : '-';
  const abiSampleRateText = Number.isFinite(abiSampleRate) ? `${Math.round(abiSampleRate * 100)}%` : '-';
  
  const abiParityState = abiParityAttempted
    ? (abiParityPass ? 'pass' : 'fail')
    : (abiPrimaryUsed ? 'primary' : 'probe');

  const statusColor = abiParityState === 'pass' ? '#24d1a0' : abiParityState === 'fail' ? '#ff5050' : '#17a3ff';
  const bgColor = abiParityState === 'pass' ? 'rgba(36,209,160,0.12)' : abiParityState === 'fail' ? 'rgba(255,80,80,0.12)' : 'rgba(23,163,255,0.12)';

  return (
    <div className="abi-gate-info">
      <div className="abi-gate-header">
        <strong>ABI Gate</strong>
        <span className="abi-parity-badge" style={{ border: `1px solid ${statusColor}`, color: statusColor, background: bgColor }}>
          {abiParityState === 'pass' ? 'parity pass' : abiParityState === 'fail' ? 'parity fail' : abiParityState === 'primary' ? 'primary' : 'probe only'}
        </span>
      </div>
      <div className="abi-gate-details">
        <div>Mode: <strong>{abiMode}</strong></div>
        <div>Sample rate: <strong>{abiSampleRateText}</strong></div>
        <div>Max abs drift: <strong>{abiParityMaxAbsText}</strong></div>
        <div>Threshold: <strong>{abiParityThresholdText}</strong></div>
        {abiProbe.parity_reason && <div>Reason: <strong>{String(abiProbe.parity_reason)}</strong></div>}
      </div>
    </div>
  );
};

export default ABIGateInfo;
