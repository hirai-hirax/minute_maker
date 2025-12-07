import { useState } from 'react'
import { MinuteGenerator } from './components/MinuteGenerator'
import { SpeakerManager } from './components/SpeakerManager'
import { Settings } from './components/Settings'
import { PromptManager } from './components/PromptManager'
import { Layout, Plus, List, Users, Settings as SettingsIcon, FileText } from 'lucide-react'
import './App.css'
import { useI18n } from './i18n'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

const TEXT = {
  ja: {
    navGenerate: '議事録作成',
    navSpeakers: '話者管理',
    navPrompts: 'プロンプト管理',
    navSettings: '設定',
    confirmReset: 'トップ画面に戻りますか？未保存の作業内容は失われます。',
    languageToggle: 'English',
    languageToggleAria: '言語を英語に切り替え',
  },
  en: {
    navGenerate: 'Minute Creation',
    navSpeakers: 'Speaker Manager',
    navPrompts: 'Prompt Manager',
    navSettings: 'Settings',
    confirmReset: 'Return to the main screen? Unsaved work will be lost.',
    languageToggle: '日本語',
    languageToggleAria: 'Switch language to Japanese',
  },
}

function App() {
  const [activeTab, setActiveTab] = useState<'generate' | 'speakers' | 'settings' | 'prompts'>('generate')
  const [resetKey, setResetKey] = useState(0)
  const { language, toggleLanguage } = useI18n()

  const text = TEXT[language]

  const handleLogoClick = () => {
    if (activeTab === 'generate') {
      if (confirm(text.confirmReset)) {
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
              {text.navGenerate}
            </button>
            <button
              className={`nav-btn ${activeTab === 'speakers' ? 'active' : ''}`}
              onClick={() => setActiveTab('speakers')}
            >
              <Users size={16} />
              {text.navSpeakers}
            </button>
            <button
              className={`nav-btn ${activeTab === 'prompts' ? 'active' : ''}`}
              onClick={() => setActiveTab('prompts')}
            >
              <FileText size={16} />
              {text.navPrompts}
            </button>
            <button
              className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveTab('settings')}
            >
              <SettingsIcon size={16} />
              {text.navSettings}
            </button>
          </nav>
          <button
            className="nav-btn"
            onClick={toggleLanguage}
            aria-label={text.languageToggleAria}
          >
            {text.languageToggle}
          </button>
        </div>
      </header>

      <main className="container main-content">
        <div className="content-wrapper" style={{ display: activeTab === 'generate' ? 'block' : 'none' }}>
          <MinuteGenerator key={resetKey} />
        </div>
        <div className="content-wrapper" style={{ display: activeTab === 'speakers' ? 'block' : 'none' }}>
          <SpeakerManager />
        </div>
        <div className="content-wrapper" style={{ display: activeTab === 'prompts' ? 'block' : 'none' }}>
          <PromptManager />
        </div>
        <div className="content-wrapper" style={{ display: activeTab === 'settings' ? 'block' : 'none' }}>
          <Settings />
        </div>
      </main>
    </div>
  )
}

export default App
