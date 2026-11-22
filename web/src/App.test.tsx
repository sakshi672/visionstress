import { afterEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

test('creates a project and loads its real API-backed workspace', async () => {
  const requests: string[] = [];
  vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
    requests.push(`${options?.method || 'GET'} ${url}`);
    if (url === '/api/projects' && options?.method === 'POST') return {ok:true,status:201,json:async()=>({id:1,name:'Drone detector',description:''})};
    if (url === '/api/projects') return {ok:true,status:200,json:async()=> requests.some(r=>r==='POST /api/projects') ? [{id:1,name:'Drone detector',description:''}] : []};
    return {ok:true,status:200,json:async()=>[]};
  }));
  render(<App />);
  fireEvent.change(screen.getByLabelText('Create project'), {target:{value:'Drone detector'}});
  fireEvent.click(screen.getByRole('button', {name:'Create'}));
  await waitFor(() => expect(screen.getByRole('option', {name:'Drone detector'})).toBeTruthy());
  expect(requests).toContain('POST /api/projects');
  expect(requests).toContain('GET /api/datasets?project_id=1');
});
