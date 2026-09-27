import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import api from './services/api'
import './App.css'

const defaultSources = [
  { file: 'README.md', start_line: 1, end_line: 8, name: 'Overview', type: 'section' },
]

function formatRepositoryLabel(value) {
  return String(value || 'Repository')
    .split(/[-_]/)
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(' ')
}

function App() {
  const [repoUrl, setRepoUrl] = useState('')
  const [localPath, setLocalPath] = useState('')
  const [repoId, setRepoId] = useState('demo-repo')
  const [question, setQuestion] = useState('')
  const [status, setStatus] = useState('Ready')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [overview, setOverview] = useState(null)
  const [graph, setGraph] = useState({ nodes: [], edges: [] })
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState(defaultSources)
  const [selectedFile, setSelectedFile] = useState('README.md')
  const [sourceContent, setSourceContent] = useState('')

  const repositoryLabel = formatRepositoryLabel(overview?.repository || repoId || 'not indexed')

  useEffect(() => {
    if (!repoId) return
    const loadOverview = async () => {
      try {
        const data = await api.getRepositoryOverview(repoId)
        setOverview(data)
      } catch {
        setOverview(null)
      }
    }

    loadOverview()
  }, [repoId])

  const loadGraph = async (id) => {
    try {
      const data = await api.getRepositoryGraph(id)
      setGraph({ nodes: data.nodes || [], edges: data.edges || [] })
    } catch {
      setGraph({ nodes: [], edges: [] })
    }
  }

  const loadSource = async (file) => {
    if (!repoId || !file) return
    try {
      const payload = await api.getRepositorySource(repoId, file)
      setSelectedFile(payload.file || file)
      setSourceContent(payload.content || '')
    } catch {
      setSourceContent('Unable to load file contents for this repository.')
    }
  }

  const handleIngest = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    setStatus('Indexing repository…')

    try {
      const payload = await api.ingestRepository({
        repo_url: repoUrl || null,
        local_path: localPath || null,
      })

      const id = payload?.stats?.repo_id || repoId
      setRepoId(id)
      setStatus('Repository indexed successfully')
      const nextOverview = await api.getRepositoryOverview(id)
      setOverview(nextOverview)
      await loadGraph(id)
      if (nextOverview?.important_files?.length) {
        await loadSource(nextOverview.important_files[0])
      }
    } catch (loadError) {
      setError(loadError.message || 'Unable to ingest repository.')
      setStatus('Ingestion failed')
    } finally {
      setLoading(false)
    }
  }

  const handleAsk = async (event) => {
    event.preventDefault()
    if (!repoId || !question.trim()) {
      setError('Enter a repository ID and a question before asking.')
      return
    }

    setLoading(true)
    setError('')
    setStatus('Generating answer…')

    try {
      const payload = await api.queryRepository({ repo_id: repoId, question, top_k: 5, debug: false })
      setAnswer(payload.answer || 'No answer generated.')
      setSources(payload.sources || defaultSources)
      const firstFile = payload.sources?.[0]?.file || selectedFile
      if (firstFile) {
        await loadSource(firstFile)
      }
      setStatus('Answer ready')
    } catch (queryError) {
      setError(queryError.message || 'Unable to generate answer.')
      setStatus('Query failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-kicker">softRAG</span>
          <h1>Repository intelligence</h1>
        </div>
        <div className="status-pill">{status}</div>
      </header>

      <main className="workspace-grid">
        <aside className="panel stack">
          <section className="section-card">
            <h2>Ingest repository</h2>
            <form className="form-stack" onSubmit={handleIngest}>
              <label>
                <span>Repository URL</span>
                <input
                  aria-label="Repository URL"
                  type="text"
                  value={repoUrl}
                  onChange={(event) => setRepoUrl(event.target.value)}
                  placeholder="https://github.com/org/repo"
                />
              </label>

              <label>
                <span>Local path</span>
                <input
                  aria-label="Local path"
                  type="text"
                  value={localPath}
                  onChange={(event) => setLocalPath(event.target.value)}
                  placeholder="Optional local repo path"
                />
              </label>

              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? 'Working…' : 'Ingest repository'}
              </button>
            </form>
          </section>

          <section className="section-card">
            <h2>Ask repository</h2>
            <form className="form-stack" onSubmit={handleAsk}>
              <label>
                <span>Repository ID</span>
                <input
                  aria-label="Repository ID"
                  type="text"
                  value={repoId}
                  onChange={(event) => setRepoId(event.target.value)}
                />
              </label>

              <label>
                <span>Question</span>
                <textarea
                  aria-label="Question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  rows={5}
                  placeholder="What routes and services are defined?"
                />
              </label>

              <button type="submit" className="secondary-button" disabled={loading}>
                {loading ? 'Thinking…' : 'Ask repository'}
              </button>
            </form>
          </section>
        </aside>

        <section className="main-column">
          {error && <div className="error-banner">{error}</div>}

          <div className="summary-grid">
            <div className="panel section-card">
              <div className="section-header">
                <h2>Repository overview</h2>
                <span className="repo-badge">{repositoryLabel}</span>
              </div>

              {overview ? (
                <div className="metric-grid">
                  <Metric label="Files" value={overview.files_processed || 0} />
                  <Metric label="Chunks" value={overview.chunks_created || 0} />
                  <Metric label="Embeddings" value={overview.embeddings_created || 0} />
                  <Metric label="Symbols" value={overview.symbol_count || 0} />
                </div>
              ) : (
                <p className="muted-text">No repository metadata loaded yet. Ingest a repo to unlock overview details.</p>
              )}

              {overview && (
                <div className="meta-stack">
                  <div>
                    <p className="eyebrow">Repository</p>
                    <p className="meta-value">{formatRepositoryLabel(overview.repository || overview.repo_id)}</p>
                  </div>
                  <div>
                    <p className="eyebrow">Languages</p>
                    <div className="tag-row">
                      {Object.entries(overview.languages || {}).map(([language, count]) => (
                        <span key={language} className="tag">{language} ({count})</span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="eyebrow">Important files</p>
                    <ul className="file-list">
                      {(overview.important_files || []).slice(0, 6).map((file) => (
                        <li key={file}>
                          <button type="button" onClick={() => loadSource(file)}>{file}</button>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>

            <div className="panel section-card">
              <h2>Dependency graph</h2>
              {graph.nodes.length ? (
                <div className="graph-block">
                  <div className="tag-row">
                    {graph.nodes.slice(0, 12).map((node) => (
                      <span key={node.id} className="tag soft-tag">{node.name || node.file}</span>
                    ))}
                  </div>
                  <p className="graph-summary">{graph.edges.length} discovered edges across {graph.nodes.length} nodes.</p>
                </div>
              ) : (
                <p className="muted-text">Graph metadata will appear once the repository is indexed.</p>
              )}
            </div>
          </div>

          <div className="panel section-card answer-card">
            <h2>Answer</h2>
            {answer ? (
              <>
                <article className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                </article>
                <div className="source-pills">
                  {sources.map((source) => (
                    <button
                      key={`${source.file}-${source.start_line || 0}`}
                      type="button"
                      onClick={() => loadSource(source.file)}
                    >
                      {source.file}:{source.start_line || 1}
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <p className="muted-text">Ask a question to generate a grounded answer from the repository index.</p>
            )}
          </div>

          <div className="panel section-card source-card">
            <div className="section-header">
              <h2>Source viewer</h2>
              <span className="source-label">{selectedFile}</span>
            </div>
            <pre className="source-viewer">
              <code>{sourceContent || 'Select a file to inspect its source.'}</code>
            </pre>
          </div>
        </section>
      </main>
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="metric-box">
      <p className="eyebrow">{label}</p>
      <p className="metric-value">{value}</p>
    </div>
  )
}

export default App
