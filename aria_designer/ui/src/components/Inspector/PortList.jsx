import React from 'react';

const PortList = ({ ports, label }) => {
  if (!ports || ports.length === 0) return null;

  return (
    <div className="port-group">
      <div className="port-group-label">{label}</div>
      {ports.map((p) => (
        <div key={p.name} className="port-row">
          <span className="port-name">{p.name}</span>
          <span className="port-dtype">{p.dtype}</span>
          {p.shape && <span className="port-shape">[{p.shape.join(', ')}]</span>}
        </div>
      ))}
    </div>
  );
};

export default PortList;
