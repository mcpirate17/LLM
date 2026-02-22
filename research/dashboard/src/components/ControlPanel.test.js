import {
  applyTelemetryPresetSettings,
  inferTelemetryPreset,
  normalizeTelemetryPresetForStorage,
} from './controlPanelTelemetryPresets';

describe('ControlPanel telemetry preset transitions', () => {
  test('applies compact preset consistently (idle/running actions share same helper)', () => {
    const next = applyTelemetryPresetSettings('compact');
    expect(next).toEqual({
      showCanarySummary: true,
      showCanaryRefreshHint: false,
      nativeTelemetryExpanded: false,
      canaryCooldownSeconds: 12,
      canaryRefreshCooldownS: 0,
      canaryTelemetryPreset: 'compact',
      canaryPrefsNotice: 'Telemetry preset set to compact.',
    });
  });

  test('applies default and debug presets with expected cooldown/detail settings', () => {
    const defaultNext = applyTelemetryPresetSettings('default');
    const debugNext = applyTelemetryPresetSettings('debug');

    expect(defaultNext.canaryCooldownSeconds).toBe(8);
    expect(defaultNext.nativeTelemetryExpanded).toBe(true);
    expect(debugNext.canaryCooldownSeconds).toBe(2);
    expect(debugNext.showCanaryRefreshHint).toBe(true);
  });

  test('rejects unknown presets', () => {
    expect(applyTelemetryPresetSettings('custom')).toBeNull();
    expect(applyTelemetryPresetSettings('')).toBeNull();
  });

  test('infers named preset and falls back to custom for divergence', () => {
    const fromDefault = applyTelemetryPresetSettings('default');
    expect(inferTelemetryPreset(fromDefault)).toBe('default');

    expect(
      inferTelemetryPreset({
        ...fromDefault,
        showCanaryRefreshHint: false,
      })
    ).toBe('custom');
  });

  test('does not persist derived custom preset', () => {
    expect(normalizeTelemetryPresetForStorage('compact')).toBe('compact');
    expect(normalizeTelemetryPresetForStorage('custom')).toBeUndefined();
  });
});
