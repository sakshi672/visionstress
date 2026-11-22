import { afterEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import SuitePanel from './SuitePanel';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

test('saves selected failure IDs and queues a candidate evaluation', async () => {
  const submitted: unknown[] = [];
  let saved = false;
  vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
    if (url === '/api/suites' && options?.method === 'POST') {
      submitted.push(JSON.parse(String(options.body)));
      saved = true;
      return { ok: true, status: 201, json: async () => ({ id: 1 }) };
    }
    if (url === '/api/suites?project_id=1') return { ok: true, status: 200, json: async () => saved ? [{ id: 1, name: 'Critical misses', snapshot: { image_ids: [7], stress: 'brightness', severity: 2, gate: { metric: 'map50_95', max_absolute_drop: 0.02 }, reference: { stressed: { map50_95: 0.4 } } }, runs: [] }] : [] };
    if (url === '/api/suites/1/runs' && options?.method === 'POST') {
      submitted.push(JSON.parse(String(options.body)));
      return { ok: true, status: 202, json: async () => ({ job_id: 9 }) };
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const onQueued = vi.fn(async () => {});
  render(<SuitePanel projectId={1} sourceJob={{ id: 3, status: 'completed', spec: { stress: 'brightness', severity: 2 }, result: { failure_examples: [{ image_id: 7, score: 3 }, { image_id: 8, score: 0 }] } }} models={[{ id: 2, name: 'yolo11s.pt' }]} jobs={[]} onQueued={onQueued}/>);
  fireEvent.change(screen.getByLabelText('Suite name'), { target: { value: 'Critical misses' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save suite' }));
  await waitFor(() => expect(screen.getByText('Critical misses')).toBeTruthy());
  expect(submitted[0]).toMatchObject({ source_job_id: 3, image_ids: [7], metric: 'map50_95' });
  fireEvent.click(screen.getByRole('button', { name: 'Run on candidate' }));
  await waitFor(() => expect(onQueued).toHaveBeenCalledOnce());
  expect(submitted[1]).toEqual({ model_id: 2 });
});
