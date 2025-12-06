import { useState } from 'react'
import { MinuteGenerator } from './components/MinuteGenerator'
import { SpeakerManager } from './components/SpeakerManager'
import { Settings } from './components/Settings'
import { Layout, Plus, List, Users, Settings as SettingsIcon } from 'lucide-react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

function App() {
  const [activeTab, setActiveTab] = useState<'generate' | 'speakers' | 'settings'>('generate')
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
              className={`nav-btn ${activeTab === 'speakers' ? 'active' : ''}`}
              onClick={() => setActiveTab('speakers')}
            >
              <Users size={16} />
              話者管理
            </button>
            <button
              className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveTab('settings')}
            >
              <SettingsIcon size={16} />
              設定
            </button>
          </nav>
        </div>
      </header>

      <main className="container main-content">
        {activeTab === 'generate' ? (
          <div className="content-wrapper">
            <MinuteGenerator key={resetKey} />
          </div>
        ) : activeTab === 'settings' ? (
          <div className="content-wrapper">
            <Settings />
          </div>
        ) : (
          <div className="content-wrapper">
            <SpeakerManager />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
