import React from 'react';

/**
 * AriaAvatar — Stylish anime-inspired SVG avatar for Dr. Aria Nexus.
 * Sleek design with gradient hair, luminous eyes, and ambient glow.
 * Mood-reactive with smooth color transitions.
 */

const MOOD_THEMES = {
  curious:        { iris: '#4a90e2', irisInner: '#74b9ff', glow: '#4a90e2', accent: '#e74c3c', hairHighlight: '#2c3e6e' },
  excited:        { iris: '#e74c3c', irisInner: '#ff7675', glow: '#e74c3c', accent: '#f1c40f', hairHighlight: '#4a1a2e' },
  contemplative:  { iris: '#a29bfe', irisInner: '#d6d3ff', glow: '#a29bfe', accent: '#6c5ce7', hairHighlight: '#2d2b55' },
  frustrated:     { iris: '#fd7272', irisInner: '#ffb8b8', glow: '#e17055', accent: '#e67e22', hairHighlight: '#4a1a1a' },
  triumphant:     { iris: '#00cec9', irisInner: '#81ecec', glow: '#00b894', accent: '#fdcb6e', hairHighlight: '#1a3c3c' },
};

function AriaAvatar({ mood = 'curious', size = 80 }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const sc = s / 80;
  const t = MOOD_THEMES[mood] || MOOD_THEMES.curious;
  const uid = `aria-${mood}-${size}`;

  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} xmlns="http://www.w3.org/2000/svg">
      <defs>
        {/* Ambient glow behind head */}
        <radialGradient id={`${uid}-glow`} cx="50%" cy="45%" r="50%">
          <stop offset="0%" stopColor={t.glow} stopOpacity="0.25" />
          <stop offset="100%" stopColor={t.glow} stopOpacity="0" />
        </radialGradient>

        {/* Hair gradient */}
        <linearGradient id={`${uid}-hair`} x1="0" y1="0" x2="0.3" y2="1">
          <stop offset="0%" stopColor="#2d3436" />
          <stop offset="40%" stopColor="#1a1a2e" />
          <stop offset="100%" stopColor={t.hairHighlight} />
        </linearGradient>

        {/* Skin gradient */}
        <radialGradient id={`${uid}-skin`} cx="50%" cy="40%" r="55%">
          <stop offset="0%" stopColor="#ffe8dc" />
          <stop offset="100%" stopColor="#f0c9b5" />
        </radialGradient>

        {/* Eye glow */}
        <radialGradient id={`${uid}-eyeglow`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={t.irisInner} stopOpacity="0.8" />
          <stop offset="60%" stopColor={t.iris} stopOpacity="1" />
          <stop offset="100%" stopColor={t.iris} stopOpacity="0.7" />
        </radialGradient>

        {/* Lip tint */}
        <linearGradient id={`${uid}-lip`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#d4666a" />
          <stop offset="100%" stopColor="#b33939" />
        </linearGradient>

        {/* Hair sheen */}
        <linearGradient id={`${uid}-sheen`} x1="0.3" y1="0" x2="0.7" y2="1">
          <stop offset="0%" stopColor="white" stopOpacity="0.12" />
          <stop offset="50%" stopColor="white" stopOpacity="0.03" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Ambient glow */}
      <circle cx={cx} cy={cy} r={36 * sc} fill={`url(#${uid}-glow)`} />

      {/* Hair back volume */}
      <ellipse cx={cx} cy={cy - 4 * sc} rx={30 * sc} ry={32 * sc}
        fill={`url(#${uid}-hair)`} />

      {/* Neck */}
      <rect x={cx - 6 * sc} y={cy + 20 * sc} width={12 * sc} height={14 * sc}
        rx={4 * sc} fill="#f0c9b5" />

      {/* Shoulder hint */}
      <ellipse cx={cx} cy={cy + 34 * sc} rx={22 * sc} ry={8 * sc}
        fill="#1a1a2e" opacity="0.9" />

      {/* Face */}
      <path
        d={`M ${cx - 19 * sc} ${cy - 10 * sc}
           Q ${cx - 22 * sc} ${cy + 8 * sc} ${cx - 13 * sc} ${cy + 20 * sc}
           Q ${cx - 6 * sc} ${cy + 25 * sc} ${cx} ${cy + 26 * sc}
           Q ${cx + 6 * sc} ${cy + 25 * sc} ${cx + 13 * sc} ${cy + 20 * sc}
           Q ${cx + 22 * sc} ${cy + 8 * sc} ${cx + 19 * sc} ${cy - 10 * sc}
           Q ${cx + 12 * sc} ${cy - 26 * sc} ${cx} ${cy - 28 * sc}
           Q ${cx - 12 * sc} ${cy - 26 * sc} ${cx - 19 * sc} ${cy - 10 * sc} Z`}
        fill={`url(#${uid}-skin)`}
      />

      {/* Cheek blush */}
      <ellipse cx={cx - 14 * sc} cy={cy + 8 * sc} rx={5 * sc} ry={3 * sc}
        fill="#ff9f9f" opacity="0.18" />
      <ellipse cx={cx + 14 * sc} cy={cy + 8 * sc} rx={5 * sc} ry={3 * sc}
        fill="#ff9f9f" opacity="0.18" />

      {/* Eyes */}
      <Eyes cx={cx} cy={cy} sc={sc} mood={mood} theme={t} uid={uid} />

      {/* Eyebrows */}
      <Eyebrows cx={cx} cy={cy} sc={sc} mood={mood} />

      {/* Nose */}
      <path
        d={`M ${cx - 0.5 * sc} ${cy + 4 * sc} Q ${cx - 2 * sc} ${cy + 10 * sc} ${cx} ${cy + 10.5 * sc}
           Q ${cx + 2 * sc} ${cy + 10 * sc} ${cx + 0.5 * sc} ${cy + 4 * sc}`}
        fill="none" stroke="#d9b0a0" strokeWidth={0.7 * sc} strokeLinecap="round" />

      {/* Mouth */}
      <Mouth cx={cx} cy={cy} sc={sc} mood={mood} uid={uid} />

      {/* Hair front - swept bangs with volume */}
      <path
        d={`M ${cx - 26 * sc} ${cy - 16 * sc}
           Q ${cx - 30 * sc} ${cy - 6 * sc} ${cx - 22 * sc} ${cy + 4 * sc}
           Q ${cx - 18 * sc} ${cy + 2 * sc} ${cx - 16 * sc} ${cy - 6 * sc}
           Q ${cx - 12 * sc} ${cy - 16 * sc} ${cx - 4 * sc} ${cy - 14 * sc}
           Q ${cx + 2 * sc} ${cy - 20 * sc} ${cx + 8 * sc} ${cy - 16 * sc}
           Q ${cx + 16 * sc} ${cy - 8 * sc} ${cx + 20 * sc} ${cy - 4 * sc}
           Q ${cx + 24 * sc} ${cy + 2 * sc} ${cx + 26 * sc} ${cy + 6 * sc}
           Q ${cx + 28 * sc} ${cy - 8 * sc} ${cx + 24 * sc} ${cy - 18 * sc}
           Q ${cx + 16 * sc} ${cy - 32 * sc} ${cx} ${cy - 34 * sc}
           Q ${cx - 16 * sc} ${cy - 32 * sc} ${cx - 26 * sc} ${cy - 16 * sc} Z`}
        fill={`url(#${uid}-hair)`}
      />

      {/* Hair sheen overlay */}
      <path
        d={`M ${cx - 26 * sc} ${cy - 16 * sc}
           Q ${cx - 30 * sc} ${cy - 6 * sc} ${cx - 22 * sc} ${cy + 4 * sc}
           Q ${cx - 18 * sc} ${cy + 2 * sc} ${cx - 16 * sc} ${cy - 6 * sc}
           Q ${cx - 12 * sc} ${cy - 16 * sc} ${cx - 4 * sc} ${cy - 14 * sc}
           Q ${cx + 2 * sc} ${cy - 20 * sc} ${cx + 8 * sc} ${cy - 16 * sc}
           Q ${cx + 16 * sc} ${cy - 8 * sc} ${cx + 20 * sc} ${cy - 4 * sc}
           Q ${cx + 24 * sc} ${cy + 2 * sc} ${cx + 26 * sc} ${cy + 6 * sc}
           Q ${cx + 28 * sc} ${cy - 8 * sc} ${cx + 24 * sc} ${cy - 18 * sc}
           Q ${cx + 16 * sc} ${cy - 32 * sc} ${cx} ${cy - 34 * sc}
           Q ${cx - 16 * sc} ${cy - 32 * sc} ${cx - 26 * sc} ${cy - 16 * sc} Z`}
        fill={`url(#${uid}-sheen)`}
      />

      {/* Wispy side strands */}
      <path d={`M ${cx - 22 * sc} ${cy + 2 * sc} Q ${cx - 26 * sc} ${cy + 14 * sc} ${cx - 22 * sc} ${cy + 24 * sc}`}
        stroke="#1a1a2e" strokeWidth={2.5 * sc} fill="none" strokeLinecap="round" />
      <path d={`M ${cx - 20 * sc} ${cy + 4 * sc} Q ${cx - 24 * sc} ${cy + 16 * sc} ${cx - 20 * sc} ${cy + 26 * sc}`}
        stroke="#2d3436" strokeWidth={1.5 * sc} fill="none" strokeLinecap="round" opacity="0.6" />
      <path d={`M ${cx + 24 * sc} ${cy + 4 * sc} Q ${cx + 28 * sc} ${cy + 16 * sc} ${cx + 24 * sc} ${cy + 28 * sc}`}
        stroke="#1a1a2e" strokeWidth={2.2 * sc} fill="none" strokeLinecap="round" />

      {/* Mood particles */}
      <MoodParticles cx={cx} cy={cy} sc={sc} mood={mood} theme={t} />
    </svg>
  );
}

function Eyes({ cx, cy, sc, mood, theme: t, uid }) {
  const ey = cy + 1 * sc;
  const lx = cx - 10 * sc;
  const rx = cx + 10 * sc;

  const eyeW = 5.5 * sc;
  const eyeH = mood === 'excited' ? 5.8 * sc : mood === 'contemplative' || mood === 'triumphant' ? 3.8 * sc : 5 * sc;

  // Upper lash thickness varies by mood
  const lashW = mood === 'excited' ? 1.8 * sc : 1.4 * sc;

  return (
    <>
      {[lx, rx].map((ex, i) => (
        <g key={i}>
          {/* Eye white with slight shadow */}
          <ellipse cx={ex} cy={ey} rx={eyeW} ry={eyeH} fill="white" />
          <ellipse cx={ex} cy={ey - eyeH * 0.3} rx={eyeW * 0.9} ry={eyeH * 0.3}
            fill="#f0e8e4" opacity="0.3" />

          {/* Iris with glow gradient */}
          <circle cx={ex} cy={ey + 0.3 * sc} r={3 * sc}
            fill={`url(#${uid}-eyeglow)`} />

          {/* Pupil */}
          <circle cx={ex} cy={ey + 0.3 * sc} r={1.4 * sc} fill="#0a0a0a" />

          {/* Catchlight - large */}
          <circle cx={ex + 1.2 * sc} cy={ey - 1 * sc} r={1 * sc} fill="white" opacity="0.9" />
          {/* Catchlight - small */}
          <circle cx={ex - 0.8 * sc} cy={ey + 1 * sc} r={0.5 * sc} fill="white" opacity="0.5" />

          {/* Upper lash line - thick and curved */}
          <path
            d={`M ${ex - eyeW} ${ey - eyeH * 0.85}
               Q ${ex} ${ey - eyeH - 1.5 * sc} ${ex + eyeW} ${ey - eyeH * 0.85}`}
            fill="none" stroke="#0f0f0f" strokeWidth={lashW} strokeLinecap="round" />

          {/* Outer lash flick */}
          <path
            d={`M ${ex + eyeW - 0.5 * sc} ${ey - eyeH * 0.7}
               Q ${ex + eyeW + 1.5 * sc} ${ey - eyeH - 1 * sc} ${ex + eyeW + 2 * sc} ${ey - eyeH - 2 * sc}`}
            fill="none" stroke="#0f0f0f" strokeWidth={0.8 * sc} strokeLinecap="round" />

          {/* Lower lash - subtle */}
          <path
            d={`M ${ex - eyeW * 0.7} ${ey + eyeH * 0.8}
               Q ${ex} ${ey + eyeH + 0.3 * sc} ${ex + eyeW * 0.7} ${ey + eyeH * 0.8}`}
            fill="none" stroke="#2d2d2d" strokeWidth={0.5 * sc} strokeLinecap="round" opacity="0.4" />

          {/* Half-lid for triumphant/contemplative */}
          {(mood === 'triumphant' || mood === 'contemplative') && (
            <ellipse cx={ex} cy={ey - eyeH * 0.15} rx={eyeW * 1.05} ry={eyeH * 0.55}
              fill={`url(#${uid}-skin)`} />
          )}
        </g>
      ))}
    </>
  );
}

function Eyebrows({ cx, cy, sc, mood }) {
  const by = cy - 10 * sc;
  const lx = cx - 10 * sc;
  const rx = cx + 10 * sc;

  let lAngle = 0;
  let rAngle = 0;
  let curve = 2 * sc;

  if (mood === 'curious') { lAngle = -2 * sc; curve = 3 * sc; }
  if (mood === 'frustrated') { lAngle = 2.5 * sc; rAngle = 1.5 * sc; curve = 1 * sc; }
  if (mood === 'excited') { lAngle = -1.5 * sc; rAngle = -1.5 * sc; curve = 2.5 * sc; }
  if (mood === 'triumphant') { lAngle = -0.5 * sc; curve = 1.5 * sc; }

  return (
    <>
      <path
        d={`M ${lx - 5 * sc} ${by + lAngle}
           Q ${lx} ${by - curve} ${lx + 5 * sc} ${by}`}
        fill="none" stroke="#1a1a2e" strokeWidth={1.6 * sc} strokeLinecap="round" />
      <path
        d={`M ${rx - 5 * sc} ${by}
           Q ${rx} ${by - curve} ${rx + 5 * sc} ${by + rAngle}`}
        fill="none" stroke="#1a1a2e" strokeWidth={1.6 * sc} strokeLinecap="round" />
    </>
  );
}

function Mouth({ cx, cy, sc, mood, uid }) {
  const my = cy + 15 * sc;

  if (mood === 'excited') {
    return (
      <g>
        <path
          d={`M ${cx - 7 * sc} ${my} Q ${cx} ${my + 6 * sc} ${cx + 7 * sc} ${my}`}
          fill="none" stroke={`url(#${uid}-lip)`} strokeWidth={1.5 * sc} strokeLinecap="round" />
        <path
          d={`M ${cx - 5 * sc} ${my + 1 * sc} Q ${cx} ${my + 4 * sc} ${cx + 5 * sc} ${my + 1 * sc}`}
          fill="#fff" opacity="0.15" />
      </g>
    );
  }

  if (mood === 'triumphant') {
    return (
      <path
        d={`M ${cx - 8 * sc} ${my + 0.5 * sc}
           Q ${cx - 3 * sc} ${my + 5 * sc} ${cx + 2 * sc} ${my + 1 * sc}
           Q ${cx + 6 * sc} ${my - 1 * sc} ${cx + 9 * sc} ${my - 2 * sc}`}
        fill="none" stroke={`url(#${uid}-lip)`} strokeWidth={1.5 * sc} strokeLinecap="round" />
    );
  }

  if (mood === 'frustrated') {
    return (
      <path
        d={`M ${cx - 6 * sc} ${my + 2 * sc}
           Q ${cx} ${my - 1 * sc} ${cx + 6 * sc} ${my + 2 * sc}`}
        fill="none" stroke="#b33939" strokeWidth={1.3 * sc} strokeLinecap="round" />
    );
  }

  if (mood === 'contemplative') {
    return (
      <path
        d={`M ${cx - 5 * sc} ${my + 1 * sc}
           Q ${cx} ${my + 0.5 * sc} ${cx + 5 * sc} ${my + 1 * sc}`}
        fill="none" stroke="#b33939" strokeWidth={1.2 * sc} strokeLinecap="round" />
    );
  }

  // Curious: elegant smirk
  return (
    <path
      d={`M ${cx - 5 * sc} ${my + 0.5 * sc}
         Q ${cx - 1 * sc} ${my + 3 * sc} ${cx + 5 * sc} ${my - 0.5 * sc}`}
      fill="none" stroke={`url(#${uid}-lip)`} strokeWidth={1.3 * sc} strokeLinecap="round" />
  );
}

function MoodParticles({ cx, cy, sc, mood, theme: t }) {
  if (mood === 'triumphant') {
    return (
      <g>
        <Sparkle x={cx + 26 * sc} y={cy - 16 * sc} size={3 * sc} color={t.accent} />
        <Sparkle x={cx - 24 * sc} y={cy - 12 * sc} size={2.5 * sc} color={t.accent} />
        <Sparkle x={cx + 18 * sc} y={cy - 26 * sc} size={2 * sc} color={t.glow} />
      </g>
    );
  }

  if (mood === 'curious') {
    return (
      <g>
        <Sparkle x={cx + 26 * sc} y={cy - 14 * sc} size={2.5 * sc} color={t.accent} />
        {/* Floating dot */}
        <circle cx={cx - 28 * sc} cy={cy - 20 * sc} r={1.2 * sc} fill={t.glow} opacity="0.5">
          <animate attributeName="cy" values={`${cy - 20 * sc};${cy - 23 * sc};${cy - 20 * sc}`}
            dur="3s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.5;0.2;0.5"
            dur="3s" repeatCount="indefinite" />
        </circle>
      </g>
    );
  }

  if (mood === 'excited') {
    return (
      <g>
        <Sparkle x={cx + 24 * sc} y={cy - 18 * sc} size={3 * sc} color={t.accent} />
        <Sparkle x={cx - 22 * sc} y={cy - 16 * sc} size={2.5 * sc} color={t.accent} />
        <Sparkle x={cx + 28 * sc} y={cy + 4 * sc} size={2 * sc} color={t.glow} />
      </g>
    );
  }

  if (mood === 'contemplative') {
    return (
      <g>
        {/* Subtle floating orbs */}
        {[[-26, -18], [28, -10]].map(([dx, dy], i) => (
          <circle key={i} cx={cx + dx * sc} cy={cy + dy * sc} r={1.5 * sc}
            fill={t.glow} opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.1;0.3"
              dur={`${3 + i}s`} repeatCount="indefinite" />
          </circle>
        ))}
      </g>
    );
  }

  return null;
}

function Sparkle({ x, y, size, color }) {
  return (
    <g opacity="0.8">
      <line x1={x} y1={y - size} x2={x} y2={y + size}
        stroke={color} strokeWidth={size * 0.3} strokeLinecap="round" />
      <line x1={x - size} y1={y} x2={x + size} y2={y}
        stroke={color} strokeWidth={size * 0.3} strokeLinecap="round" />
      <line x1={x - size * 0.55} y1={y - size * 0.55} x2={x + size * 0.55} y2={y + size * 0.55}
        stroke={color} strokeWidth={size * 0.2} strokeLinecap="round" />
      <line x1={x + size * 0.55} y1={y - size * 0.55} x2={x - size * 0.55} y2={y + size * 0.55}
        stroke={color} strokeWidth={size * 0.2} strokeLinecap="round" />
    </g>
  );
}

export default AriaAvatar;
