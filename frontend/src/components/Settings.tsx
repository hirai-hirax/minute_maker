import React, { useState, useEffect } from 'react'
import { Save, RotateCcw, CheckCircle2 } from 'lucide-react'
import './Settings.css'

interface AppSettings {
    llmProvider: 'azure' | 'ollama'
    ollamaBaseUrl: string
    ollamaModel: string
    summaryModel: 'gpt-4o' | 'gpt-4o-mini'
    whisperProvider: 'azure' | 'faster-whisper'
    whisperModel: 'gpt-4o' | 'whisper'
    ossWhisperModel: 'tiny' | 'base' | 'small' | 'medium' | 'large-v2' | 'large-v3'
}

const DEFAULT_SETTINGS: AppSettings = {
    llmProvider: 'azure',
    ollamaBaseUrl: 'http://localhost:11434/v1',
    ollamaModel: 'llama3.1',
    summaryModel: 'gpt-4o',
    whisperProvider: 'azure',
    whisperModel: 'gpt-4o',
    ossWhisperModel: 'base'
}

const AZURE_SUMMARY_MODELS: { value: AppSettings['summaryModel']; label: string; description: string }[] = [
    {
        value: 'gpt-4o',
        label: 'GPT-4o',
        description: '高品質な要約を高精度で生成'
    },
    {
        value: 'gpt-4o-mini',
        label: 'GPT-4o mini',
        description: '軽量・高速でコストを抑えた生成'
    }
]

export function Settings() {
    const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS)
    const [saveSuccess, setSaveSuccess] = useState(false)

    useEffect(() => {
        // Load settings from localStorage
        try {
            const stored = localStorage.getItem('minute-maker-settings')
            if (stored) {
                setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(stored) })
            }
        } catch (error) {
            console.error('Failed to load settings:', error)
        }
    }, [])

    const handleSave = () => {
        try {
            localStorage.setItem('minute-maker-settings', JSON.stringify(settings))
            setSaveSuccess(true)
            setTimeout(() => setSaveSuccess(false), 3000)
        } catch (error) {
            console.error('Failed to save settings:', error)
            alert('設定の保存に失敗しました')
        }
    }

    const handleReset = () => {
        if (confirm('設定をデフォルトに戻しますか？')) {
            setSettings(DEFAULT_SETTINGS)
            localStorage.removeItem('minute-maker-settings')
            setSaveSuccess(true)
            setTimeout(() => setSaveSuccess(false), 3000)
        }
    }

    return (
        <div className="settings-container">
            <div className="settings-header">
                <h2>設定</h2>
                <p className="settings-subtitle">要約用のLLMモデルと文字起こしモデルを選択してください</p>
            </div>

            {/* LLM Provider Settings */}
            <div className="settings-section">
                <h3>LLMプロバイダー（要約生成）</h3>
                <div className="setting-group">
                    <label className="radio-label">
                        <input
                            type="radio"
                            name="llmProvider"
                            value="azure"
                            checked={settings.llmProvider === 'azure'}
                            onChange={(e) => setSettings({ ...settings, llmProvider: e.target.value as 'azure' })}
                        />
                        <div className="radio-content">
                            <span className="radio-title">Azure OpenAI</span>
                            <span className="radio-description">クラウドベースのGPT-4o（高品質・高速）</span>
                        </div>
                    </label>

                    {settings.llmProvider === 'azure' && (
                        <div className="nested-settings">
                            <div className="input-group">
                                <label htmlFor="summaryModel">要約モデル</label>
                                <select
                                    id="summaryModel"
                                    value={settings.summaryModel}
                                    onChange={(e) => setSettings({ ...settings, summaryModel: e.target.value as AppSettings['summaryModel'] })}
                                >
                                    {AZURE_SUMMARY_MODELS.map(model => (
                                        <option key={model.value} value={model.value}>
                                            {model.label}（{model.description}）
                                        </option>
                                    ))}
                                </select>
                                <small>要約に使用するAzure OpenAIのモデルを選択します</small>
                            </div>
                        </div>
                    )}

                    <label className="radio-label">
                        <input
                            type="radio"
                            name="llmProvider"
                            value="ollama"
                            checked={settings.llmProvider === 'ollama'}
                            onChange={(e) => setSettings({ ...settings, llmProvider: e.target.value as 'ollama' })}
                        />
                        <div className="radio-content">
                            <span className="radio-title">Ollama（オンプレミス）</span>
                            <span className="radio-description">ローカル実行（プライバシー重視・オフライン対応）</span>
                        </div>
                    </label>

                    {settings.llmProvider === 'ollama' && (
                        <div className="nested-settings">
                            <div className="input-group">
                                <label htmlFor="ollamaBaseUrl">Ollama URL</label>
                                <input
                                    type="text"
                                    id="ollamaBaseUrl"
                                    value={settings.ollamaBaseUrl}
                                    onChange={(e) => setSettings({ ...settings, ollamaBaseUrl: e.target.value })}
                                    placeholder="http://localhost:11434/v1"
                                />
                            </div>
                            <div className="input-group">
                                <label htmlFor="ollamaModel">モデル名</label>
                                <input
                                    type="text"
                                    id="ollamaModel"
                                    value={settings.ollamaModel}
                                    onChange={(e) => setSettings({ ...settings, ollamaModel: e.target.value })}
                                    placeholder="llama3.1, qwen2.5:7b, など"
                                />
                                <small>Ollamaにインストール済みのモデル名を入力してください</small>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Whisper Model Settings */}
            <div className="settings-section">
                <h3>文字起こしモデル</h3>
                <div className="setting-group">
                    <label className="radio-label">
                        <input
                            type="radio"
                            name="whisperProvider"
                            value="azure-gpt4o"
                            checked={settings.whisperProvider === 'azure' && settings.whisperModel === 'gpt-4o'}
                            onChange={() => setSettings({ ...settings, whisperProvider: 'azure', whisperModel: 'gpt-4o' })}
                        />
                        <div className="radio-content">
                            <span className="radio-title">GPT-4o Transcribe & Diarize</span>
                            <span className="radio-description">文字起こし + 話者分離（最高品質・Azure専用）</span>
                        </div>
                    </label>

                    <label className="radio-label">
                        <input
                            type="radio"
                            name="whisperProvider"
                            value="azure-whisper"
                            checked={settings.whisperProvider === 'azure' && settings.whisperModel === 'whisper'}
                            onChange={() => setSettings({ ...settings, whisperProvider: 'azure', whisperModel: 'whisper' })}
                        />
                        <div className="radio-content">
                            <span className="radio-title">Whisper（Azure OpenAI）</span>
                            <span className="radio-description">標準的な文字起こし（話者分離なし）</span>
                        </div>
                    </label>

                    <label className="radio-label">
                        <input
                            type="radio"
                            name="whisperProvider"
                            value="faster-whisper"
                            checked={settings.whisperProvider === 'faster-whisper'}
                            onChange={() => setSettings({ ...settings, whisperProvider: 'faster-whisper' })}
                        />
                        <div className="radio-content">
                            <span className="radio-title">Faster Whisper（OSS・オンプレミス）</span>
                            <span className="radio-description">ローカル実行・高速・無料</span>
                        </div>
                    </label>

                    {settings.whisperProvider === 'faster-whisper' && (
                        <div className="nested-settings">
                            <div className="input-group">
                                <label htmlFor="ossWhisperModel">モデルサイズ</label>
                                <select
                                    id="ossWhisperModel"
                                    value={settings.ossWhisperModel}
                                    onChange={(e) => setSettings({ ...settings, ossWhisperModel: e.target.value as any })}
                                >
                                    <option value="tiny">Tiny（最速・低精度）</option>
                                    <option value="base">Base（高速・標準精度）</option>
                                    <option value="small">Small（バランス）</option>
                                    <option value="medium">Medium（高精度・やや遅い）</option>
                                    <option value="large-v2">Large v2（最高精度・遅い）</option>
                                    <option value="large-v3">Large v3（最新・最高精度）</option>
                                </select>
                                <small>モデルサイズが大きいほど精度が向上しますが、処理時間も増加します</small>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Action Buttons */}
            <div className="settings-actions">
                <button className="btn-secondary" onClick={handleReset}>
                    <RotateCcw size={16} />
                    デフォルトに戻す
                </button>
                <button className="btn-primary" onClick={handleSave}>
                    <Save size={16} />
                    保存
                </button>
            </div>

            {/* Success Message */}
            {saveSuccess && (
                <div className="save-success">
                    <CheckCircle2 size={20} />
                    設定を保存しました
                </div>
            )}
        </div>
    )
}
