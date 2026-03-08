import React, { useState, useMemo } from 'react';
import { formatNum } from '../../utils/format';

const OpProfileTable = ({ opProfiles: initialOpProfiles }) => {
  const [sortCol, setSortCol] = useState('flops');
  const [sortAsc, setSortAsc] = useState(false);

  const opProfiles = useMemo(() => {
    const sorted = [...initialOpProfiles].sort((a, b) => {
      const av = a[sortCol] ?? 0;
      const bv = b[sortCol] ?? 0;
      return sortAsc ? av - bv : bv - av;
    });
    return sorted;
  }, [initialOpProfiles, sortCol, sortAsc]);

  const handleSort = (col) => {
    if (sortCol === col) setSortAsc(!sortAsc);
    else { setSortCol(col); setSortAsc(false); }
  };

  const sortDirection = (col) => (sortCol === col ? (sortAsc ? 'ascending' : 'descending') : 'none');
  const sortGlyph = (col) => (sortCol === col ? (sortAsc ? '▲' : '▼') : '');

  if (opProfiles.length === 0) return null;

  return (
    <div style={{ maxHeight: 200, overflowY: 'auto' }}>
      <table className="op-profile-table">
        <thead>
          <tr>
            <th aria-sort={sortDirection('op_name')}>
              <button type="button" className="th-sort-btn" onClick={() => handleSort('op_name')}>
                Op {sortGlyph('op_name')}
              </button>
            </th>
            <th aria-sort={sortDirection('flops')}>
              <button type="button" className="th-sort-btn" onClick={() => handleSort('flops')}>
                FLOPs {sortGlyph('flops')}
              </button>
            </th>
            <th aria-sort={sortDirection('params')}>
              <button type="button" className="th-sort-btn" onClick={() => handleSort('params')}>
                Params {sortGlyph('params')}
              </button>
            </th>
            <th aria-sort={sortDirection('memory_bytes')}>
              <button type="button" className="th-sort-btn" onClick={() => handleSort('memory_bytes')}>
                Mem {sortGlyph('memory_bytes')}
              </button>
            </th>
            <th>K</th>
          </tr>
        </thead>
        <tbody>
          {opProfiles.map((op, i) => (
            <tr key={i}>
              <td>{op.op_name}</td>
              <td>{formatNum(op.flops)}</td>
              <td>{formatNum(op.params)}</td>
              <td>{formatNum(op.memory_bytes)}</td>
              <td>{op.has_native_kernel ? <span className="native-badge">C</span> : null}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default OpProfileTable;
