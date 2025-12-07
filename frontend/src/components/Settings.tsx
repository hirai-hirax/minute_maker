import React, { useState, useEffect, useMemo } from 'react'
import { Save, RotateCcw, CheckCircle2 } from 'lucide-react'
import './Settings.css'
import { useI18n } from '../i18n'

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

const TEXT = {
    ja: {
        title: '設定',
        subtitle: '要約用のLLMモデルと文字起こしモデルを選択してください',
        llmProvider: 'LLMプロバイダー（要約生成）',
        azureTitle: 'Azure OpenAI',
        azureDescription: 'クラウドベースのGPT-4o（高品質・高速）',
        summaryModel: '要約モデル',
        summaryModelHelp: '要約に使用するAzure OpenAIのモデルを選択します',
        azureModels: {
            'gpt-4o': '高品質な要約を高精度で生成',
            'gpt-4o-mini': '軽量・高速でコストを抑えた生成',
        },
        ollamaTitle: 'Ollama（オンプレミス）',
        ollamaDescription: 'ローカル実行（プライバシー重視・オフライン対応）',
        ollamaUrl: 'Ollama URL',
        ollamaModel: 'モデル名',
        ollamaModelPlaceholder: 'llama3.1, qwen2.5:7b, など',
        ollamaModelHelp: 'Ollamaにインストール済みのモデル名を入力してください',
        whisperTitle: '文字起こしモデル',
        gpt4oTranscribe: 'GPT-4o Transcribe & Diarize',
        gpt4oTranscribeDesc: '文字起こし + 話者分離（最高品質・Azure専用）',
        azureWhisperTitle: 'Whisper（Azure OpenAI）',
        azureWhisperDesc: '標準的な文字起こし（話者分離なし）',
        fasterWhisperTitle: 'Faster Whisper（OSS・オンプレミス）',
        fasterWhisperDesc: 'ローカル実行・高速・無料',
        ossModelLabel: 'モデルサイズ',
        ossModelOptions: {
            tiny: 'Tiny（最速・低精度）',
            base: 'Base（高速・標準精度）',
            small: 'Small（バランス）',
            medium: 'Medium（高精度・やや遅い）',
            'large-v2': 'Large v2（最高精度・遅い）',
            'large-v3': 'Large v3（最新・最高精度）',
        },
        ossModelHelp: 'モデルサイズが大きいほど精度が向上しますが、処理時間も増加します',
        reset: 'デフォルトに戻す',
        save: '保存',
        saveSuccess: '設定を保存しました',
        saveFail: '設定の保存に失敗しました',
        resetConfirm: '設定をデフォルトに戻しますか？',
    },
    en: {
        title: 'Settings',
        subtitle: 'Choose the LLM and transcription models for summarization.',
        llmProvider: 'LLM Provider (Summarization)',
        azureTitle: 'Azure OpenAI',
        azureDescription: 'Cloud-based GPT-4o (high quality, fast)',
        summaryModel: 'Summary model',
        summaryModelHelp: 'Select the Azure OpenAI model used for summarization',
        azureModels: {
            'gpt-4o': 'High-accuracy summaries with GPT-4o',
            'gpt-4o-mini': 'Lightweight, fast, and cost efficient',
        },
        ollamaTitle: 'Ollama (on-premise)',
        ollamaDescription: 'Runs locally for privacy and offline use',
        ollamaUrl: 'Ollama URL',
        ollamaModel: 'Model name',
        ollamaModelPlaceholder: 'llama3.1, qwen2.5:7b, etc.',
        ollamaModelHelp: 'Enter a model already installed in Ollama',
        whisperTitle: 'Transcription models',
        gpt4oTranscribe: 'GPT-4o Transcribe & Diarize',
        gpt4oTranscribeDesc: 'Transcription with speaker separation (Azure only)',
        azureWhisperTitle: 'Whisper (Azure OpenAI)',
        azureWhisperDesc: 'Standard transcription (no diarization)',
        fasterWhisperTitle: 'Faster Whisper (OSS / on-premise)',
        fasterWhisperDesc: 'Runs locally, fast, and free',
        ossModelLabel: 'Model size',
        ossModelOptions: {
            tiny: 'Tiny (fastest, lower accuracy)',
            base: 'Base (fast, standard accuracy)',
            small: 'Small (balanced)',
            medium: 'Medium (higher accuracy, slower)',
            'large-v2': 'Large v2 (highest accuracy, slower)',
            'large-v3': 'Large v3 (latest, highest accuracy)',
        },
        ossModelHelp: 'Larger models improve accuracy but take longer to run.',
        reset: 'Reset to default',
        save: 'Save',
        saveSuccess: 'Settings saved',
        saveFail: 'Failed to save settings',
        resetConfirm: 'Reset settings to default?',
    }
} as const

export function Settings() {
    const { language } = useI18n()
    const text = TEXT[language]

    const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS)
    const [saveSuccess, setSaveSuccess] = useState(false)

    const azureSummaryModels = useMemo(
        () => [
            {
                value: 'gpt-4o' as const,
                label: 'GPT-4o',
                description: text.azureModels['gpt-4o'],
            },
            {
                value: 'gpt-4o-mini' as const,
                label: 'GPT-4o mini',
                description: text.azureModels['gpt-4o-mini'],
            },
        ],
        [text]
    )

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
            alert(text.saveFail)
        }
    }

    const handleReset = () => {
        if (confirm(text.resetConfirm)) {
            setSettings(DEFAULT_SETTINGS)
            localStorage.removeItem('minute-maker-settings')
            setSaveSuccess(true)
            setTimeout(() => setSaveSuccess(false), 3000)
        }
    }

    return (
        <div className="settings-container">
            <div className="settings-header">
                <h2>{text.title}</h2>
                <p className="settings-subtitle">{text.subtitle}</p>
            </div>

            {/* LLM Provider Settings */}
            <div className="settings-section">
                <h3>{text.llmProvider}</h3>
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
                            <span className="radio-title">{text.azureTitle}</span>
                            <span className="radio-description">{text.azureDescription}</span>
                        </div>
                    </label>

                    {settings.llmProvider === 'azure' && (
                        <div className="nested-settings">
                            <div className="input-group">
                                <label htmlFor="summaryModel">{text.summaryModel}</label>
                                <select
                                    id="summaryModel"
                                    value={settings.summaryModel}
                                    onChange={(e) => setSettings({ ...settings, summaryModel: e.target.value as AppSettings['summaryModel'] })}
                                >
                                    {azureSummaryModels.map(model => (
                                        <option key={model.value} value={model.value}>
                                            {model.label}（{model.description}）
                                        </option>
                                    ))}
                                </select>
                                <small>{text.summaryModelHelp}</small>
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
                            <span className="radio-title">{text.ollamaTitle}</span>
                            <span className="radio-description">{text.ollamaDescription}</span>
                        </div>
                    </label>

                    {settings.llmProvider === 'ollama' && (
                        <div className="nested-settings">
                            <div className="input-group">
                                <label htmlFor="ollamaBaseUrl">{text.ollamaUrl}</label>
                                <input
                                    type="text"
                                    id="ollamaBaseUrl"
                                    value={settings.ollamaBaseUrl}
                                    onChange={(e) => setSettings({ ...settings, ollamaBaseUrl: e.target.value })}
                                    placeholder="http://localhost:11434/v1"
                                />
                            </div>
                            <div className="input-group">
                                <label htmlFor="ollamaModel">{text.ollamaModel}</label>
                                <input
                                    type="text"
                                    id="ollamaModel"
                                    value={settings.ollamaModel}
                                    onChange={(e) => setSettings({ ...settings, ollamaModel: e.target.value })}
                                    placeholder={text.ollamaModelPlaceholder}
                                />
                                <small>{text.ollamaModelHelp}</small>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Whisper Model Settings */}
            <div className="settings-section">
                <h3>{text.whisperTitle}</h3>
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
                            <span className="radio-title">{text.gpt4oTranscribe}</span>
                            <span className="radio-description">{text.gpt4oTranscribeDesc}</span>
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
                            <span className="radio-title">{text.azureWhisperTitle}</span>
                            <span className="radio-description">{text.azureWhisperDesc}</span>
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
                            <span className="radio-title">{text.fasterWhisperTitle}</span>
                            <span className="radio-description">{text.fasterWhisperDesc}</span>
                        </div>
                    </label>

                    {settings.whisperProvider === 'faster-whisper' && (
                        <div className="nested-settings">
                            <div className="input-group">
                                <label htmlFor="ossWhisperModel">{text.ossModelLabel}</label>
                                <select
                                    id="ossWhisperModel"
                                    value={settings.ossWhisperModel}
                                    onChange={(e) => setSettings({ ...settings, ossWhisperModel: e.target.value as AppSettings['ossWhisperModel'] })}
                                >
                                    <option value="tiny">{text.ossModelOptions.tiny}</option>
                                    <option value="base">{text.ossModelOptions.base}</option>
                                    <option value="small">{text.ossModelOptions.small}</option>
                                    <option value="medium">{text.ossModelOptions.medium}</option>
                                    <option value="large-v2">{text.ossModelOptions['large-v2']}</option>
                                    <option value="large-v3">{text.ossModelOptions['large-v3']}</option>
                                </select>
                                <small>{text.ossModelHelp}</small>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Action Buttons */}
            <div className="settings-actions">
                <button className="btn-secondary" onClick={handleReset}>
                    <RotateCcw size={16} />
                    {text.reset}
                </button>
                <button className="btn-primary" onClick={handleSave}>
                    <Save size={16} />
                    {text.save}
                </button>
            </div>

            {/* Success Message */}
            {saveSuccess && (
                <div className="save-success">
                    <CheckCircle2 size={20} />
                    {text.saveSuccess}
                </div>
            )}
        </div>
    )
}
