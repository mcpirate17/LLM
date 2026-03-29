import React from 'react';
import { formatNum } from '../../utils/format';
import useSortableRows from './useSortableRows';

const OpProfileTable = ({
  opProfiles: initialOpProfiles,
  maxHeight = 200,
  actionHint = null,
}) => {
  const {
    rows: opProfiles,
    handleSort,
    sortDirection,
    sortGlyph,
  } = useSortableRows(initialOpProfiles, 'flops', false);

  if (opProfiles.length === 0) return null;

  return (
    <>
      {actionHint && (
        <div style={{ marginBottom: 8, fontSize: 11, color: 'var(--muted)', lineHeight: 1.45 }}>
          {actionHint}
        </div>
      )}
      <div className="table-scroll-shell" style={{ maxHeight }}>
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
            {opProfiles.map((op) => (
              <tr key={op.op_name}>
                <td style={{ whiteSpace: 'normal', overflowWrap: 'anywhere', lineHeight: 1.35 }}>{op.op_name}</td>
                <td>{formatNum(op.flops)}</td>
                <td>{formatNum(op.params)}</td>
                <td>{formatNum(op.memory_bytes)}</td>
                <td>{op.has_native_kernel ? <span className="native-badge">C</span> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
};

export default OpProfileTable;
