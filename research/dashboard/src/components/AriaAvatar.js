import React from 'react';

/**
 * AriaAvatar — Procedural anime-style SVG avatar for Dr. Aria Nexus.
 *
 * Mood-reactive expressions:
 *   curious:        raised eyebrow, small smile, sparkle
 *   excited:        wide eyes, big grin, blush marks
 *   contemplative:  half-closed eyes, neutral mouth
 *   frustrated:     furrowed brows, slight frown
 *   triumphant:     happy closed eyes, wide grin, sparkles
 */

const MOOD_COLORS = {
  curious:        { iris: '#58a6ff', accent: '#bc8cff' },
  excited:        { iris: '#3fb950', accent: '#f0883e' },
  contemplative:  { iris: '#8b949e', accent: '#58a6ff' },
  frustrated:     { iris: '#f85149', accent: '#d29922' },
  triumphant:     { iris: '#d29922', accent: '#3fb950' },
};

function AriaAvatar({ mood = 'curious', size = 80 }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const scale = s / 80; // base design is 80x80
  const colors = MOOD_COLORS[mood] || MOOD_COLORS.curious;

  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} xmlns="http://www.w3.org/2000/svg">
      {/* Hair back */}
      <ellipse cx={cx} cy={cy - 2 * scale} rx={32 * scale} ry={34 * scale}
        fill="#2d1b69" />

      {/* Face */}
      <ellipse cx={cx} cy={cy + 2 * scale} rx={24 * scale} ry={26 * scale}
        fill="#fce4d6" />

      {/* Hair front - bangs */}
      <path d={`M ${cx - 24 * scale} ${cy - 8 * scale}
                Q ${cx - 18 * scale} ${cy - 28 * scale} ${cx - 6 * scale} ${cy - 18 * scale}
                Q ${cx} ${cy - 22 * scale} ${cx + 6 * scale} ${cy - 18 * scale}
                Q ${cx + 18 * scale} ${cy - 28 * scale} ${cx + 24 * scale} ${cy - 8 * scale}
                Q ${cx + 20 * scale} ${cy - 20 * scale} ${cx} ${cy - 26 * scale}
                Q ${cx - 20 * scale} ${cy - 20 * scale} ${cx - 24 * scale} ${cy - 8 * scale}
                Z`}
        fill="#3d2b7a" />

      {/* Hair side strands */}
      <path d={`M ${cx - 24 * scale} ${cy - 6 * scale}
                Q ${cx - 30 * scale} ${cy + 10 * scale} ${cx - 26 * scale} ${cy + 22 * scale}`}
        stroke="#3d2b7a" strokeWidth={4 * scale} fill="none" strokeLinecap="round" />
      <path d={`M ${cx + 24 * scale} ${cy - 6 * scale}
                Q ${cx + 30 * scale} ${cy + 10 * scale} ${cx + 26 * scale} ${cy + 22 * scale}`}
        stroke="#3d2b7a" strokeWidth={4 * scale} fill="none" strokeLinecap="round" />

      {/* Eyes */}
      <Eyes cx={cx} cy={cy} scale={scale} mood={mood} colors={colors} />

      {/* Blush marks (excited/triumphant) */}
      {(mood === 'excited' || mood === 'triumphant') && (
        <>
          <ellipse cx={cx - 16 * scale} cy={cy + 8 * scale} rx={4 * scale} ry={2.5 * scale}
            fill="#f8a4a4" opacity={0.5} />
          <ellipse cx={cx + 16 * scale} cy={cy + 8 * scale} rx={4 * scale} ry={2.5 * scale}
            fill="#f8a4a4" opacity={0.5} />
        </>
      )}

      {/* Nose */}
      <path d={`M ${cx} ${cy + 4 * scale} l ${2 * scale} ${4 * scale} l ${-4 * scale} 0 Z`}
        fill="none" stroke="#dbb8a0" strokeWidth={scale * 0.8} />

      {/* Mouth */}
      <Mouth cx={cx} cy={cy} scale={scale} mood={mood} />

      {/* Glasses */}
      <circle cx={cx - 10 * scale} cy={cy + 2 * scale} r={8 * scale}
        fill="none" stroke="#8b949e" strokeWidth={scale} />
      <circle cx={cx + 10 * scale} cy={cy + 2 * scale} r={8 * scale}
        fill="none" stroke="#8b949e" strokeWidth={scale} />
      <line x1={cx - 2 * scale} y1={cy + 2 * scale}
            x2={cx + 2 * scale} y2={cy + 2 * scale}
        stroke="#8b949e" strokeWidth={scale} />

      {/* Lab coat collar hint */}
      <path d={`M ${cx - 16 * scale} ${cy + 26 * scale}
                L ${cx - 6 * scale} ${cy + 20 * scale}
                L ${cx} ${cy + 24 * scale}
                L ${cx + 6 * scale} ${cy + 20 * scale}
                L ${cx + 16 * scale} ${cy + 26 * scale}`}
        fill="none" stroke="#e6edf3" strokeWidth={1.5 * scale} strokeLinejoin="round" />

      {/* Sparkles (curious/triumphant) */}
      {(mood === 'curious' || mood === 'triumphant') && (
        <>
          <Sparkle x={cx + 22 * scale} y={cy - 16 * scale} size={3 * scale} color={colors.accent} />
          {mood === 'triumphant' && (
            <Sparkle x={cx - 20 * scale} y={cy - 14 * scale} size={2.5 * scale} color={colors.accent} />
          )}
        </>
      )}

      {/* Frustrated brow crease */}
      {mood === 'frustrated' && (
        <line x1={cx - 2 * scale} y1={cy - 10 * scale}
              x2={cx + 2 * scale} y2={cy - 10 * scale}
          stroke="#dbb8a0" strokeWidth={scale * 1.2} strokeLinecap="round" />
      )}
    </svg>
  );
}

function Eyes({ cx, cy, scale, mood, colors }) {
  const eyeY = cy + 2 * scale;
  const leftX = cx - 10 * scale;
  const rightX = cx + 10 * scale;
  const eyeRx = 4.5 * scale;
  const eyeRy = mood === 'excited' ? 5 * scale : 4 * scale;

  if (mood === 'triumphant') {
    // Happy closed eyes - curved lines
    return (
      <>
        <path d={`M ${leftX - eyeRx} ${eyeY} Q ${leftX} ${eyeY - 3 * scale} ${leftX + eyeRx} ${eyeY}`}
          fill="none" stroke="#3d2b7a" strokeWidth={1.5 * scale} strokeLinecap="round" />
        <path d={`M ${rightX - eyeRx} ${eyeY} Q ${rightX} ${eyeY - 3 * scale} ${rightX + eyeRx} ${eyeY}`}
          fill="none" stroke="#3d2b7a" strokeWidth={1.5 * scale} strokeLinecap="round" />
      </>
    );
  }

  if (mood === 'contemplative') {
    // Half-closed eyes
    return (
      <>
        <ellipse cx={leftX} cy={eyeY + scale} rx={eyeRx} ry={2.5 * scale} fill="white" />
        <ellipse cx={leftX} cy={eyeY + scale} rx={2.5 * scale} ry={2 * scale} fill={colors.iris} />
        <ellipse cx={leftX} cy={eyeY + scale} rx={1.2 * scale} ry={1 * scale} fill="#1a1a2e" />
        {/* Eyelid */}
        <path d={`M ${leftX - eyeRx - scale} ${eyeY - scale}
                  Q ${leftX} ${eyeY + scale} ${leftX + eyeRx + scale} ${eyeY - scale}`}
          fill="#fce4d6" stroke="none" />

        <ellipse cx={rightX} cy={eyeY + scale} rx={eyeRx} ry={2.5 * scale} fill="white" />
        <ellipse cx={rightX} cy={eyeY + scale} rx={2.5 * scale} ry={2 * scale} fill={colors.iris} />
        <ellipse cx={rightX} cy={eyeY + scale} rx={1.2 * scale} ry={1 * scale} fill="#1a1a2e" />
        <path d={`M ${rightX - eyeRx - scale} ${eyeY - scale}
                  Q ${rightX} ${eyeY + scale} ${rightX + eyeRx + scale} ${eyeY - scale}`}
          fill="#fce4d6" stroke="none" />
      </>
    );
  }

  // Default eyes (curious, excited, frustrated)
  const browOffset = mood === 'curious' ? -2 * scale : 0;
  const rightBrowOffset = mood === 'curious' ? 0 : (mood === 'frustrated' ? 2 * scale : 0);

  return (
    <>
      {/* Eyebrows */}
      <line x1={leftX - eyeRx} y1={eyeY - eyeRy - 2 * scale + browOffset}
            x2={leftX + eyeRx} y2={eyeY - eyeRy - 2 * scale}
        stroke="#3d2b7a" strokeWidth={1.2 * scale} strokeLinecap="round" />
      <line x1={rightX - eyeRx} y1={eyeY - eyeRy - 2 * scale + rightBrowOffset}
            x2={rightX + eyeRx} y2={eyeY - eyeRy - 2 * scale + (mood === 'frustrated' ? -rightBrowOffset : 0)}
        stroke="#3d2b7a" strokeWidth={1.2 * scale} strokeLinecap="round" />

      {/* Eye whites */}
      <ellipse cx={leftX} cy={eyeY} rx={eyeRx} ry={eyeRy} fill="white" />
      <ellipse cx={rightX} cy={eyeY} rx={eyeRx} ry={eyeRy} fill="white" />

      {/* Iris */}
      <circle cx={leftX} cy={eyeY} r={2.5 * scale} fill={colors.iris} />
      <circle cx={rightX} cy={eyeY} r={2.5 * scale} fill={colors.iris} />

      {/* Pupils */}
      <circle cx={leftX} cy={eyeY} r={1.2 * scale} fill="#1a1a2e" />
      <circle cx={rightX} cy={eyeY} r={1.2 * scale} fill="#1a1a2e" />

      {/* Eye highlights */}
      <circle cx={leftX + scale} cy={eyeY - scale} r={0.8 * scale} fill="white" opacity={0.8} />
      <circle cx={rightX + scale} cy={eyeY - scale} r={0.8 * scale} fill="white" opacity={0.8} />
    </>
  );
}

function Mouth({ cx, cy, scale, mood }) {
  const mouthY = cy + 14 * scale;

  if (mood === 'excited' || mood === 'triumphant') {
    // Big grin
    const width = mood === 'triumphant' ? 10 : 8;
    return (
      <path d={`M ${cx - width * scale} ${mouthY}
                Q ${cx} ${mouthY + 6 * scale} ${cx + width * scale} ${mouthY}`}
        fill="none" stroke="#c0392b" strokeWidth={1.5 * scale} strokeLinecap="round" />
    );
  }

  if (mood === 'frustrated') {
    // Slight frown
    return (
      <path d={`M ${cx - 6 * scale} ${mouthY + 2 * scale}
                Q ${cx} ${mouthY - 2 * scale} ${cx + 6 * scale} ${mouthY + 2 * scale}`}
        fill="none" stroke="#c0392b" strokeWidth={1.2 * scale} strokeLinecap="round" />
    );
  }

  if (mood === 'contemplative') {
    // Neutral line
    return (
      <line x1={cx - 5 * scale} y1={mouthY}
            x2={cx + 5 * scale} y2={mouthY}
        stroke="#c0392b" strokeWidth={1.2 * scale} strokeLinecap="round" />
    );
  }

  // Curious: small smile
  return (
    <path d={`M ${cx - 5 * scale} ${mouthY}
              Q ${cx} ${mouthY + 3 * scale} ${cx + 5 * scale} ${mouthY}`}
      fill="none" stroke="#c0392b" strokeWidth={1.2 * scale} strokeLinecap="round" />
  );
}

function Sparkle({ x, y, size, color }) {
  return (
    <g>
      <line x1={x} y1={y - size} x2={x} y2={y + size}
        stroke={color} strokeWidth={size * 0.3} strokeLinecap="round" />
      <line x1={x - size} y1={y} x2={x + size} y2={y}
        stroke={color} strokeWidth={size * 0.3} strokeLinecap="round" />
      <line x1={x - size * 0.6} y1={y - size * 0.6}
            x2={x + size * 0.6} y2={y + size * 0.6}
        stroke={color} strokeWidth={size * 0.2} strokeLinecap="round" />
      <line x1={x + size * 0.6} y1={y - size * 0.6}
            x2={x - size * 0.6} y2={y + size * 0.6}
        stroke={color} strokeWidth={size * 0.2} strokeLinecap="round" />
    </g>
  );
}

export default AriaAvatar;
