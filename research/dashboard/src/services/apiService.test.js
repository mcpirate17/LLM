import { apiService } from './apiService';

describe('apiService rerun queue actions', () => {
  let originalFetch;

  beforeEach(() => {
    originalFetch = global.fetch;
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ status: 'launched' }),
    }));
  });

  afterEach(() => {
    global.fetch = originalFetch;
    jest.restoreAllMocks();
  });

  it('gives manual queue drain a runner-scale timeout', async () => {
    const timeoutSpy = jest.spyOn(global, 'setTimeout');

    await apiService.drainPendingValidationRerun();

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/runner/drain-pending-validation-rerun',
      expect.objectContaining({ method: 'POST' })
    );
    expect(timeoutSpy).toHaveBeenCalledWith(expect.any(Function), 15 * 60 * 1000);
  });
});
