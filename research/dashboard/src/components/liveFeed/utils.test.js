import { curveSegmentLabel, splitCurveIntoSegments } from './utils';

describe('live feed loss curve segmentation', () => {
  test('splits by explicit candidate and training program metadata', () => {
    const curve = [
      { step: 0, loss: 2.0, source_result_id: 'a', candidate_index: 1, total_candidates: 2, training_program_index: 1, total_training_programs: 3 },
      { step: 10, loss: 1.8, source_result_id: 'a', candidate_index: 1, total_candidates: 2, training_program_index: 1, total_training_programs: 3 },
      { step: 10, loss: 1.7, source_result_id: 'a', candidate_index: 1, total_candidates: 2, training_program_index: 2, total_training_programs: 3 },
      { step: 20, loss: 1.5, source_result_id: 'a', candidate_index: 1, total_candidates: 2, training_program_index: 2, total_training_programs: 3 },
      { step: 20, loss: 1.6, source_result_id: 'b', candidate_index: 2, total_candidates: 2, training_program_index: 1, total_training_programs: 3 },
    ];

    const segments = splitCurveIntoSegments(curve);

    expect(segments).toHaveLength(3);
    expect(segments.map((segment, index) => curveSegmentLabel(segment, 'program', index))).toEqual([
      'c1/2 p1/3',
      'c1/2 p2/3',
      'c2/2 p1/3',
    ]);
  });

  test('keeps legacy step-reset splitting as a fallback', () => {
    const segments = splitCurveIntoSegments([
      { step: 0, loss: 2.0 },
      { step: 10, loss: 1.8 },
      { step: 0, loss: 2.1 },
      { step: 10, loss: 1.9 },
    ]);

    expect(segments).toHaveLength(2);
  });
});
