const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }

  return response.status === 204 ? null : response.json()
}

export const api = {
  healthCheck: () => request('/health'),

  ingestRepository: ({ repo_url, local_path }) =>
    request('/ingest', {
      method: 'POST',
      body: JSON.stringify({ repo_url, local_path }),
    }),

  queryRepository: ({ repo_id, question, top_k = 5, debug = false }) =>
    request('/query', {
      method: 'POST',
      body: JSON.stringify({ repo_id, question, top_k, debug }),
    }),

  getRepositoryOverview: (repo_id) =>
    request(`/repository/${encodeURIComponent(repo_id)}/overview`),

  getRepositoryGraph: (repo_id) =>
    request(`/repository/${encodeURIComponent(repo_id)}/graph`),

  getRepositorySource: (repo_id, file) =>
    request(`/repository/${encodeURIComponent(repo_id)}/source?file=${encodeURIComponent(file)}`),
}

export default api
