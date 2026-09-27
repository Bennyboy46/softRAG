import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import api from './services/api'

vi.mock('./services/api', () => ({
  default: {
    ingestRepository: vi.fn(),
    queryRepository: vi.fn(),
    getRepositoryOverview: vi.fn(),
    getRepositoryGraph: vi.fn(),
    getRepositorySource: vi.fn(),
  },
}))

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('ingests a repository and shows overview stats', async () => {
    api.ingestRepository.mockResolvedValue({
      status: 'success',
      message: 'Indexed',
      stats: {
        repo_id: 'demo-repo',
        repository: 'demo-repo',
        files_processed: 6,
        chunks_created: 12,
        embeddings_created: 12,
        languages: { python: 4 },
      },
    })

    api.getRepositoryOverview.mockResolvedValue({
      repo_id: 'demo-repo',
      repository: 'demo-repo',
      files_processed: 6,
      chunks_created: 12,
      embeddings_created: 12,
      languages: { python: 4 },
      important_files: ['app.py'],
      api_routes: ['health'],
      symbol_count: 8,
    })

    api.getRepositoryGraph.mockResolvedValue({
      nodes: [
        { id: 'app.py', file: 'app.py', type: 'module', name: 'app' },
      ],
      edges: [],
    })

    api.getRepositorySource.mockResolvedValue({
      file: 'app.py',
      language: 'python',
      content: 'def health():\n    return "ok"\n',
    })

    render(<App />)

    fireEvent.change(screen.getByLabelText(/repository url/i), {
      target: { value: 'https://github.com/example/demo' },
    })

    fireEvent.click(screen.getByRole('button', { name: /ingest repository/i }))

    await waitFor(() => {
      expect(api.ingestRepository).toHaveBeenCalledTimes(1)
    })

    expect((await screen.findAllByText(/Demo repo/i)).length).toBeGreaterThan(0)
  })

  it('submits a repository question and shows the answer', async () => {
    api.queryRepository.mockResolvedValue({
      answer: 'The app exposes a /health route.',
      sources: [{ file: 'app.py', start_line: 1, end_line: 3, name: 'health', type: 'function' }],
    })

    render(<App />)

    fireEvent.change(screen.getByLabelText(/repository id/i), {
      target: { value: 'demo-repo' },
    })
    fireEvent.change(screen.getByLabelText(/question/i), {
      target: { value: 'What endpoints exist?' },
    })

    fireEvent.click(screen.getByRole('button', { name: /ask repository/i }))

    await waitFor(() => {
      expect(api.queryRepository).toHaveBeenCalledTimes(1)
    })

    expect(await screen.findByText(/The app exposes a \/health route\./i)).toBeInTheDocument()
  })
})
