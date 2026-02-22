export const TELEMETRY_PRESET_KEYS = ['compact', 'default', 'debug'];

export const TELEMETRY_PRESETS = {
  compact: {
    showCanarySummary: true,
    showCanaryRefreshHint: false,
    nativeTelemetryExpanded: false,
    canaryCooldownSeconds: 12,
  },
  default: {
    showCanarySummary: true,
    showCanaryRefreshHint: true,
    nativeTelemetryExpanded: true,
    canaryCooldownSeconds: 8,
  },
  debug: {
    showCanarySummary: true,
    showCanaryRefreshHint: true,
    nativeTelemetryExpanded: true,
    canaryCooldownSeconds: 2,
  },
};

export const clampCanaryCooldown = (value) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return 8;
  return Math.max(0, Math.min(60, Math.round(num)));
};

export const inferTelemetryPreset = ({
  showCanarySummary,
  showCanaryRefreshHint,
  nativeTelemetryExpanded,
  canaryCooldownSeconds,
}) => {
  const cooldown = clampCanaryCooldown(canaryCooldownSeconds);
  const found = TELEMETRY_PRESET_KEYS.find((key) => {
    const preset = TELEMETRY_PRESETS[key];
    return (
      preset.showCanarySummary === Boolean(showCanarySummary)
      && preset.showCanaryRefreshHint === Boolean(showCanaryRefreshHint)
      && preset.nativeTelemetryExpanded === Boolean(nativeTelemetryExpanded)
      && preset.canaryCooldownSeconds === cooldown
    );
  });
  return found || 'custom';
};

export const normalizeTelemetryPresetForStorage = (preset) => {
  return TELEMETRY_PRESET_KEYS.includes(preset) ? preset : undefined;
};

export const applyTelemetryPresetSettings = (preset) => {
  if (!TELEMETRY_PRESET_KEYS.includes(preset)) {
    return null;
  }
  const values = TELEMETRY_PRESETS[preset];
  return {
    showCanarySummary: values.showCanarySummary,
    showCanaryRefreshHint: values.showCanaryRefreshHint,
    nativeTelemetryExpanded: values.nativeTelemetryExpanded,
    canaryCooldownSeconds: values.canaryCooldownSeconds,
    canaryRefreshCooldownS: 0,
    canaryTelemetryPreset: preset,
    canaryPrefsNotice: `Telemetry preset set to ${preset}.`,
  };
};
