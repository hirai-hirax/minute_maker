import { useState } from 'react'
import { MinuteGenerator } from './components/MinuteGenerator'
import { Layout, Plus, List } from 'lucide-react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

function App() {
  const [activeTab, setActiveTab] = useState<'generate' | 'list'>('generate')
  const [resetKey, setResetKey] = useState(0)

  const handleLogoClick = () => {
    if (activeTab === 'generate') {
      if (confirm('トップ画面に戻りますか？未保存の作業内容は失われます。')) {
        setResetKey(prev => prev + 1)
      }
    } else {
      setActiveTab('generate')
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="container header-content">
          <div className="logo-section" onClick={handleLogoClick} style={{ cursor: 'pointer' }}>
            <div className="logo-icon">
              <Layout size={20} color="white" />
            </div>
            <div>
              <h1 className="app-title">Minute Maker AI</h1>
            </div>
          </div>
          <nav className="nav-tabs">
            <button
              className={`nav-btn ${activeTab === 'generate' ? 'active' : ''}`}
              onClick={() => setActiveTab('generate')}
            >
              <Plus size={16} />
              作成
            </button>
            <button
              className={`nav-btn ${activeTab === 'list' ? 'active' : ''}`}
              onClick={() => setActiveTab('list')}
            >
              <List size={16} />
              一覧
            </button>
          </nav>
        </div>
      </header>

      <main className="container main-content">
        {activeTab === 'generate' ? (
          <div className="content-wrapper">
            <MinuteGenerator key={resetKey} />
          </div>
        ) : (
          <div className="empty-state">
            <p>議事録一覧機能は現在メンテナンス中です。</p>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
