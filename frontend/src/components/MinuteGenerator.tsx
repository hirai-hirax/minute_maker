import React, { useState, useRef } from 'react'
import { Upload, FileAudio, CheckCircle, Loader2, FileText, FileSpreadsheet, Play, Pause, UserPlus, X, RefreshCw, ArrowRight, Paperclip } from 'lucide-react'
import { useI18n } from '../i18n'
import './MinuteGenerator.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

type ProcessState = 'idle' | 'uploading' | 'processing' | 'reviewing' | 'summarizing_setup' | 'summarizing' | 'completed' | 'error';

interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    speaker: string;
}

interface PromptPreset {
    id: string;
    name: string;
    description: string;
}

interface ProcessingResult {
    id: string;
    transcript: string;
    segments: TranscriptSegment[];
    summary?: string;
    speakers: string[];
    action_items?: string[];
    decisions?: string[];
    [key: string]: any;  // カスタムフィールドを許可
}

const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

const getSettings = () => {
    const defaults = {
        llmProvider: 'azure',
        ollamaBaseUrl: 'http://localhost:11434/v1',
        ollamaModel: 'llama3.1',
        summaryModel: 'gpt-4o',
        whisperProvider: 'azure',
        whisperModel: 'gpt-4o',
        ossWhisperModel: 'base'
    }

    try {
        const stored = localStorage.getItem('minute-maker-settings')
        return stored ? { ...defaults, ...JSON.parse(stored) } : defaults
    } catch {
        return defaults
    }
}

const STRINGS = {
    ja: {
        customFieldNote: 'カスタムフィールドを許可',
        downloadTitlePrefix: '議事録',
        registerModalTitle: '話者登録',
        registerModalDescription: (name: string) => `以下のセグメントの音声を使用して、話者「${name}」の特徴をシステムに登録します。`,
        registerModalNameLabel: '話者名',
        registerModalNamePlaceholder: '例: 山田太郎',
        cancel: 'キャンセル',
        register: '登録する',
        eyebrow: 'AI 自動生成',
        heroTitle: '音声・動画から議事録を作成',
        dropLabel: 'ファイルをドラッグ＆ドロップ',
        dropHelp: 'または クリックして選択',
        dropFormats: 'MP3, WAV, MP4, M4A 対応',
        startButton: '生成を開始する',
        stepLabels: { processing: '文字起こし', reviewing: '確認・編集', summarizing_setup: '要約・整形', completed: '完了' },
        restartConfirm: '現在の作業内容は破棄されます。最初からやり直しますか？',
        transcriptTitle: '文字起こし結果',
        identifySpeakers: '話者識別を実行',
        goToSummary: '要約・整形の確認へ',
        registerSpeakerTitle: 'この話者を登録',
        summarySettingsTitle: '要約生成の設定',
        mergedTranscriptLabel: '整形済み文字起こし',
        mergedTranscriptDescription: '確認用に複数行をまとめたテキスト形式を切り替えできます。',
        viewList: 'リスト表示',
        viewText: 'テキスト表示',
        transcriptTextAria: '整形済み文字起こしのテキスト表示',
        promptSelect: 'プロンプト選択',
        promptSelectDescription: '目的に合わせた要約スタイルを選択してください。',
        referenceMaterials: '参考資料',
        referenceUploadLabel: '要約に使用する参考資料をアップロード (任意)',
        chooseFile: 'ファイルを選択',
        removeTitle: '削除',
        runSummary: 'AI要約を実行',
        back: '戻る',
        summaryTitle: '要約結果',
        overview: '概要',
        decisions: '決定事項',
        actionItems: 'アクションアイテム',
        processingAudio: '音声処理中...',
        generatingSummary: 'AIが要約を作成中...',
        downloadFailed: (msg: string) => `ダウンロードに失敗しました: ${msg}`,
        identifyComplete: '話者識別が完了しました！',
        identifyFailed: '話者識別に失敗しました。',
        summarizeFailed: '要約に失敗しました。',
        registerSuccess: (name: string) => `話者「${name}」を登録しました。「話者識別を実行」を押して反映してください。`,
        registrationError: (msg: string) => `エラー: ${msg}`,
        unknownSpeaker: 'Unknown',
        registerSpeaker: '話者を登録',
    },
    en: {
        customFieldNote: 'Allow custom fields',
        downloadTitlePrefix: 'minutes',
        registerModalTitle: 'Register Speaker',
        registerModalDescription: (name: string) => `Register the characteristics of "${name}" using the following segment.`,
        registerModalNameLabel: 'Speaker name',
        registerModalNamePlaceholder: 'e.g., Taro Yamada',
        cancel: 'Cancel',
        register: 'Register',
        eyebrow: 'AI Generated',
        heroTitle: 'Create meeting minutes from audio or video',
        dropLabel: 'Drag & drop a file',
        dropHelp: 'or click to choose',
        dropFormats: 'Supports MP3, WAV, MP4, M4A',
        startButton: 'Start generating',
        stepLabels: { processing: 'Transcribe', reviewing: 'Review & edit', summarizing_setup: 'Summarize', completed: 'Complete' },
        restartConfirm: 'Current progress will be discarded. Do you want to start over?',
        transcriptTitle: 'Transcription results',
        identifySpeakers: 'Identify speakers',
        goToSummary: 'Proceed to summary setup',
        registerSpeakerTitle: 'Register this speaker',
        summarySettingsTitle: 'Summary settings',
        mergedTranscriptLabel: 'Structured transcript',
        mergedTranscriptDescription: 'Switch views of the merged transcript for review.',
        viewList: 'List view',
        viewText: 'Text view',
        transcriptTextAria: 'Text view of structured transcript',
        promptSelect: 'Prompt selection',
        promptSelectDescription: 'Choose the summarization style.',
        referenceMaterials: 'Reference files',
        referenceUploadLabel: 'Upload optional reference files for the summary',
        chooseFile: 'Choose files',
        removeTitle: 'Remove',
        runSummary: 'Run AI summary',
        back: 'Back',
        summaryTitle: 'Summary results',
        overview: 'Overview',
        decisions: 'Decisions',
        actionItems: 'Action items',
        processingAudio: 'Processing audio...',
        generatingSummary: 'Generating summary...',
        downloadFailed: (msg: string) => `Failed to download: ${msg}`,
        identifyComplete: 'Speaker identification complete!',
        identifyFailed: 'Failed to identify speakers.',
        summarizeFailed: 'Summarization failed.',
        registerSuccess: (name: string) => `Speaker "${name}" registered successfully. Click "Identify speakers" to apply changes.`,
        registrationError: (msg: string) => `Error: ${msg}`,
        unknownSpeaker: 'Unknown',
        registerSpeaker: 'Register speaker',
    }
} as const

export function MinuteGenerator() {
    const { language } = useI18n()
    const text = STRINGS[language]

    const [state, setState] = useState<ProcessState>('idle');
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<ProcessingResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [prompts, setPrompts] = useState<PromptPreset[]>([]);
    const [selectedPromptId, setSelectedPromptId] = useState<string>('standard');
    const [mergedTranscript, setMergedTranscript] = useState<{ speaker: string; text: string }[]>([]);
    const [transcriptViewMode, setTranscriptViewMode] = useState<'structured' | 'text'>('structured');
    const [referenceFiles, setReferenceFiles] = useState<File[]>([]);

    const [registerModal, setRegisterModal] = useState<{ isOpen: boolean; segment: TranscriptSegment | null; name: string }>({
        isOpen: false,
        segment: null,
        name: ''
    });
    const fileInputRef = useRef<HTMLInputElement>(null);
    const referenceInputRef = useRef<HTMLInputElement>(null);

    React.useEffect(() => {
        fetch(`${API_BASE}/api/prompts`)
            .then(res => res.json())
            .then(data => setPrompts(data))
            .catch(console.error);
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const startProcess = async () => {
        if (!file) return;

        setState('uploading');
        setError(null);

        const settings = getSettings();
        const formData = new FormData();
        formData.append('file', file);

        // Send model selection based on settings
        if (settings.whisperProvider === 'azure') {
            formData.append('model', settings.whisperModel); // 'gpt-4o' or 'whisper'
        } else {
            formData.append('model', 'faster-whisper');
            formData.append('oss_model', settings.ossWhisperModel);
        }

        try {
            setState('processing'); // Combined transcribing/diarizing visual state

            const response = await fetch(`${API_BASE}/api/process_audio`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Processing failed');

            const data = await response.json();
            setResult(data);
            setState('reviewing'); // Stop here for review
        } catch (err) {
            console.error(err);
            setError(err instanceof Error ? err.message : 'An unknown error occurred');
            setState('error');
        }
    };

    const handleIdentifySpeakers = async () => {
        if (!result) return;
        setState('processing');

        const formData = new FormData();
        formData.append('process_id', result.id);
        formData.append('transcript_json', JSON.stringify(result.segments));

        try {
            const response = await fetch(`${API_BASE}/api/identify_speakers`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Identification failed');

            const data = await response.json();
            setResult({ ...result, segments: data.segments, speakers: data.speakers, transcript: data.transcript });
            setState('reviewing');
            alert(text.identifyComplete);
        } catch (err) {
            console.error(err);
            alert(text.identifyFailed);
            setState('reviewing');
        }
    };

    const prepareSummarySetup = () => {
        if (!result) return;

        // Merge segments by speaker
        const merged: { speaker: string; text: string }[] = [];
        let currentSpeaker = "";
        let currentText: string[] = [];

        result.segments.forEach((seg) => {
            const speakerName = seg.speaker || text.unknownSpeaker

            if (speakerName !== currentSpeaker) {
                if (currentText.length > 0) {
                    merged.push({
                        speaker: currentSpeaker,
                        text: currentText.join(" ")
                    });
                }
                currentSpeaker = speakerName;
                currentText = [seg.text];
            } else {
                currentText.push(seg.text);
            }
        });
        if (currentText.length > 0) {
            merged.push({ speaker: currentSpeaker, text: currentText.join(" ") });
        }

        setMergedTranscript(merged);
        setTranscriptViewMode('structured');
        setState('summarizing_setup');
    };

    const formattedTranscriptText = React.useMemo(() => {
        return mergedTranscript
            .map((item) => `${item.speaker || text.unknownSpeaker}: ${item.text}`)
            .join("\n\n");
    }, [mergedTranscript, text.unknownSpeaker]);

    const handleReferenceFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            const filesArray = Array.from(e.target.files);
            setReferenceFiles(prev => [...prev, ...filesArray]);
        }
    };

    const removeReferenceFile = (index: number) => {
        setReferenceFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleSummarize = async () => {
        if (!result) return;
        setState('summarizing');

        const transcriptText = mergedTranscript.map(m => `${m.speaker}: ${m.text}`).join("\n");
        const settings = getSettings();

        try {
            // Use FormData to support file uploads
            const formData = new FormData();
            formData.append('transcript', transcriptText);
            formData.append('prompt_id', selectedPromptId);
            formData.append('llm_provider', settings.llmProvider);

            if (settings.llmProvider === 'ollama') {
                formData.append('ollama_base_url', settings.ollamaBaseUrl);
                formData.append('ollama_model', settings.ollamaModel);
            } else {
                formData.append('azure_model', settings.summaryModel);
            }

            // Append reference files
            referenceFiles.forEach(file => {
                formData.append('reference_files', file);
            });
            const response = await fetch(`${API_BASE}/api/generate_summary`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Summarization failed');

            const summaryData = await response.json();
            setResult({
                ...result,
                ...summaryData  // すべてのフィールドを含める（カスタムフィールド含む）
            });
            setState('completed');
        } catch (err) {
            console.error(err);
            alert(text.summarizeFailed);
            setState('summarizing_setup');
        }
    };

    const handleRegisterSpeaker = async () => {
        if (!registerModal.segment || !result || !registerModal.name.trim()) return;

        try {
            const response = await fetch(`${API_BASE}/api/register_speaker`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    process_id: result.id,
                    start: registerModal.segment.start,
                    end: registerModal.segment.end,
                    speaker_name: registerModal.name
                }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Registration failed');
            }

            alert(text.registerSuccess(registerModal.name));
            setRegisterModal({ isOpen: false, segment: null, name: '' });
        } catch (e: any) {
            alert(text.registrationError(e.message));
        }
    };

    const openRegisterModal = (segment: TranscriptSegment) => {
        setRegisterModal({
            isOpen: true,
            segment: segment,
            name: segment.speaker || ''
        });
    };

    const downloadFile = async (type: 'docx' | 'xlsx') => {
        if (!result) return;

        try {
            const dateLocale = language === 'ja' ? 'ja-JP' : 'en-US'
            const response = await fetch(`${API_BASE}/api/download_minutes?format=${type}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: `${text.downloadTitlePrefix}_${new Date().toLocaleDateString(dateLocale).replace(/\//g, '-')}`,
                    summary: result.summary || '',
                    decisions: result.decisions || [],
                    action_items: result.action_items || [],
                    segments: result.segments,
                    // カスタムフィールドも含める
                    ...Object.keys(result).reduce((acc, key) => {
                        if (!['id', 'transcript', 'segments', 'summary', 'speakers', 'action_items', 'decisions'].includes(key)) {
                            acc[key] = result[key];
                        }
                        return acc;
                    }, {} as Record<string, any>)
                }),
            });

            if (!response.ok) throw new Error('Download failed');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${text.downloadTitlePrefix}_${new Date().toISOString().split('T')[0]}.${type}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            console.error(err);
            alert(text.downloadFailed(err instanceof Error ? err.message : 'Unknown error'));
        }
    };

    return (
        <div className="minute-generator animate-fade-in">
            {registerModal.isOpen && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <div className="modal-header">
                            <h3>{text.registerModalTitle}</h3>
                            <button onClick={() => setRegisterModal({ ...registerModal, isOpen: false })}>
                                <X size={20} />
                            </button>
                        </div>
                        <div className="modal-body">
                            <p className="text-sm text-secondary mb-4">
                                {text.registerModalDescription(registerModal.name)}
                            </p>
                            <div className="segment-preview">
                                "{registerModal.segment?.text}"
                            </div>
                            <label className="input-label mt-4">{text.registerModalNameLabel}</label>
                            <input
                                type="text"
                                className="input-field"
                                value={registerModal.name}
                                onChange={(e) => setRegisterModal({ ...registerModal, name: e.target.value })}
                                placeholder={text.registerModalNamePlaceholder}
                                autoFocus
                            />
                        </div>
                        <div className="modal-footer">
                            <button className="btn btn-secondary" onClick={() => setRegisterModal({ ...registerModal, isOpen: false })}>
                                {text.cancel}
                            </button>
                            <button className="btn btn-primary" onClick={handleRegisterSpeaker}>
                                {text.register}
                            </button>
                        </div>
                    </div>
                </div>
            )}
            <div className="card">
                <div className="card__header mb-4">
                    <div>
                        <p className="eyebrow text-accent-primary">{text.eyebrow}</p>
                        <h2>{text.heroTitle}</h2>
                    </div>
                </div>

                {state === 'idle' && (
                    <div
                        className="upload-zone"
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            hidden
                            accept="audio/*,video/*"
                            onChange={handleFileSelect}
                        />
                        <div className="upload-icon">
                            {file ? <FileAudio size={48} /> : <Upload size={48} />}
                        </div>
                        <h3>{file ? file.name : text.dropLabel}</h3>
                        <p className="text-secondary mt-4">{text.dropHelp}</p>
                        <p className="text-sm text-secondary mt-2 mb-6">{text.dropFormats}</p>


                        {file && (
                            <button
                                className="btn btn-primary mt-8"
                                onClick={(e) => { e.stopPropagation(); startProcess(); }}
                            >
                                {text.startButton}
                            </button>
                        )}
                    </div>
                )}

                {state !== 'idle' && state !== 'error' && (
                    <div className="process-view">
                        <div className="progress-steps">
                            {['processing', 'reviewing', 'summarizing_setup', 'completed'].map((step, idx) => {
                                const stepLabels: any = text.stepLabels;
                                let visualState = state;
                                if (state === 'summarizing') visualState = 'summarizing_setup';
                                if (state === 'uploading') visualState = 'processing';

                                const isActive = visualState === step;
                                let isCompleted = false;

                                if (visualState === 'completed') {
                                    isCompleted = step !== 'completed';
                                } else if (visualState === 'summarizing_setup') {
                                    isCompleted = step === 'processing' || step === 'reviewing';
                                } else if (visualState === 'reviewing') {
                                    isCompleted = step === 'processing';
                                }

                                // Navigation Logic
                                    const isClickable = (() => {
                                        if (state === 'uploading' || state === 'processing' || state === 'summarizing') return false;

                                    // Target step check
                                    if (step === 'processing') return true;
                                    if (step === 'reviewing') return !!result;
                                    if (step === 'summarizing_setup') return !!result && mergedTranscript.length > 0;
                                    if (step === 'completed') return !!result && !!result.summary;
                                    return false;
                                })();

                                    const handleStepClick = () => {
                                        if (!isClickable) return;
                                        if (step === 'processing') {
                                            if (confirm(text.restartConfirm)) {
                                                setState('idle');
                                                setFile(null);
                                                setResult(null);
                                        }
                                        return;
                                    }
                                    if (step === 'reviewing') setState('reviewing');
                                    if (step === 'summarizing_setup') setState('summarizing_setup');
                                    if (step === 'completed') setState('completed');
                                };

                                return (
                                    <div
                                        key={step}
                                        className={`step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''} ${isClickable ? 'cursor-pointer hover:opacity-80' : ''}`}
                                        onClick={handleStepClick}
                                    >
                                        <div className="step-icon">
                                            {isCompleted ? <CheckCircle size={20} /> : isActive ? <Loader2 size={20} className="animate-spin" /> : <div style={{ width: 20, height: 20 }} >{idx + 1}</div>}
                                        </div>
                                        <span className="step-label">{stepLabels[step]}</span>
                                    </div>
                                );
                            })}
                        </div>

                        {/* Transcript Review Screen */}
                        {state === 'reviewing' && result && (
                            <div className="result-area animate-fade-in">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="font-medium">{text.transcriptTitle}</h3>
                                    <div className="flex gap-2">
                                        <button className="btn btn-secondary" onClick={handleIdentifySpeakers}>
                                            <RefreshCw size={16} /> {text.identifySpeakers}
                                        </button>
                                        <button className="btn btn-primary" onClick={prepareSummarySetup}>
                                            {text.goToSummary} <ArrowRight size={16} />
                                        </button>
                                    </div>
                                </div>
                                <div className="transcript-box">
                                    <table className="transcript-table">
                                        <thead>
                                            <tr>
                                                <th className="w-time">Time</th>
                                                <th className="w-speaker">Speaker</th>
                                                <th>Content</th>
                                                <th className="w-action"></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {result.segments.map((seg, idx) => (
                                                <tr key={idx}>
                                                    <td className="cell-time">
                                                        <span className="time-badge">
                                                            {formatTime(seg.start)} - {formatTime(seg.end)}
                                                        </span>
                                                    </td>
                                                    <td className="cell-speaker">
                                                        <div className="speaker-badge">
                                                            {seg.speaker || text.unknownSpeaker}
                                                        </div>
                                                    </td>
                                                    <td className="cell-text">{seg.text}</td>
                                                    <td className="cell-action">
                                                        <button
                                                            className="icon-btn"
                                                            title={text.registerSpeakerTitle}
                                                            onClick={() => openRegisterModal(seg)}
                                                        >
                                                            <UserPlus size={18} />
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}

                        {/* Summary Setup Screen */}
                        {(state === 'summarizing_setup' || state === 'summarizing') && (
                            <div className="result-area animate-fade-in">
                                {state === 'summarizing' ? (
                                    <div className="flex flex-col items-center justify-center p-12 text-secondary">
                                        <Loader2 size={48} className="animate-spin mb-4 text-primary" />
                                        <p>{text.generatingSummary}</p>
                                    </div>
                                ) : (
                                    <>
                                        <h3 className="font-medium mb-4">{text.summarySettingsTitle}</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                            <div className="col-span-2">
                                                <div className="flex items-start justify-between gap-4 mb-2 flex-wrap">
                                                    <div>
                                                        <h4 className="text-sm font-medium text-secondary">{text.mergedTranscriptLabel}</h4>
                                                        <p className="text-xs text-secondary mt-1">{text.mergedTranscriptDescription}</p>
                                                    </div>
                                                    <div className="view-toggle">
                                                        <button
                                                            className={`toggle-btn ${transcriptViewMode === 'structured' ? 'active' : ''}`}
                                                            onClick={() => setTranscriptViewMode('structured')}
                                                            type="button"
                                                        >
                                                            {text.viewList}
                                                        </button>
                                                        <button
                                                            className={`toggle-btn ${transcriptViewMode === 'text' ? 'active' : ''}`}
                                                            onClick={() => setTranscriptViewMode('text')}
                                                            type="button"
                                                        >
                                                            {text.viewText}
                                                        </button>
                                                    </div>
                                                </div>
                                                {transcriptViewMode === 'structured' ? (
                                                    <div className="transcript-box" style={{ maxHeight: '400px' }}>
                                                        {mergedTranscript.map((item, idx) => (
                                                            <div key={idx} className="mb-4 pb-2 border-b border-gray-700 last:border-0">
                                                                <div className="font-bold text-accent-primary text-sm mb-1">{item.speaker || text.unknownSpeaker}</div>
                                                                <div className="text-sm leading-relaxed">{item.text}</div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <textarea
                                                        className="transcript-textarea"
                                                        value={formattedTranscriptText}
                                                        readOnly
                                                        aria-label={text.transcriptTextAria}
                                                    />
                                                )}
                                            </div>
                                            <div className="col-span-1">
                                                <div className="bg-secondary p-4 rounded-lg border border-border">
                                                    <h4 className="font-medium mb-4">{text.promptSelect}</h4>
                                                    <p className="text-secondary text-sm mb-4">
                                                        {text.promptSelectDescription}
                                                    </p>
                                                    <div className="flex flex-col gap-3">
                                                        {prompts.map(p => (
                                                            <label key={p.id} className={`p-3 rounded border cursor-pointer transition-all ${selectedPromptId === p.id ? 'border-accent-primary bg-primary bg-opacity-10' : 'border-border hover:border-text-secondary'}`}>
                                                                <input
                                                                    type="radio"
                                                                    name="prompt"
                                                                    value={p.id}
                                                                    checked={selectedPromptId === p.id}
                                                                    onChange={() => setSelectedPromptId(p.id)}
                                                                    className="mr-2"
                                                                />
                                                                <span className="font-medium">{p.name}</span>
                                                                <div className="text-xs text-secondary mt-1 pl-5">{p.description}</div>
                                                            </label>
                                                        ))}
                                                    </div>
                                                    {/* Reference Materials Section */}
                                                    <div className="mt-6 pt-4 border-t border-border">
                                                        <h4 className="font-medium mb-2 flex items-center gap-2">
                                                            <Paperclip size={18} />
                                                            {text.referenceMaterials}
                                                        </h4>
                                                        <p className="text-secondary text-xs mb-3">
                                                            {text.referenceUploadLabel}
                                                        </p>

                                                        <input
                                                            type="file"
                                                            ref={referenceInputRef}
                                                            hidden
                                                            accept=".docx,.xlsx,.pptx,.pdf,.txt"
                                                            multiple
                                                            onChange={handleReferenceFileSelect}
                                                        />

                                                        <button
                                                            className="btn btn-secondary w-full mb-3 text-sm"
                                                            onClick={() => referenceInputRef.current?.click()}
                                                            type="button"
                                                        >
                                                            <Paperclip size={16} />
                                                            {text.chooseFile}
                                                        </button>

                                                        {referenceFiles.length > 0 && (
                                                            <div className="reference-files-list">
                                                                {referenceFiles.map((file, idx) => (
                                                                    <div key={idx} className="reference-file-item">
                                                                        <span className="file-name">{file.name}</span>
                                                                        <button
                                                                            className="remove-btn"
                                                                            onClick={() => removeReferenceFile(idx)}
                                                                            type="button"
                                                                            title={text.removeTitle}
                                                                        >
                                                                            <X size={14} />
                                                                        </button>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>



                                                    <button className="btn btn-primary w-full mt-6" onClick={handleSummarize}>
                                                        {text.runSummary}
                                                    </button>

                                                    <button className="btn btn-secondary w-full mt-2" onClick={() => setState('reviewing')}>
                                                        {text.back}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        )}

                        {/* Summary / Completed Screen */}
                        {state === 'completed' && result && (
                            <div className="result-area animate-fade-in">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="font-medium">{text.summaryTitle}</h3>
                                    <div className="actions">
                                        <button className="btn btn-secondary" onClick={() => downloadFile('docx')}>
                                            <FileText size={18} /> Word
                                        </button>
                                        <button className="btn btn-secondary" onClick={() => downloadFile('xlsx')}>
                                            <FileSpreadsheet size={18} /> Excel
                                        </button>
                                        <button className="btn btn-secondary" onClick={() => setState('reviewing')}>
                                            {text.back}
                                        </button>
                                    </div>
                                </div>

                                <div className="summary-section animate-fade-in">
                                    {result.summary && (
                                        <>
                                            <h4>{text.overview}</h4>
                                            <div className="summary-box mb-4">
                                                {result.summary}
                                            </div>
                                        </>
                                    )}

                                    {(result.decisions && result.decisions.length > 0) && (
                                        <>
                                            <h4>{text.decisions}</h4>
                                            <ul className="summary-list">
                                                {result.decisions.map((d, i) => <li key={i}>{d}</li>)}
                                            </ul>
                                        </>
                                    )}

                                    {(result.action_items && result.action_items.length > 0) && (
                                        <>
                                            <h4>{text.actionItems}</h4>
                                            <ul className="summary-list">
                                                {result.action_items.map((i, idx) => <li key={idx}>{i}</li>)}
                                            </ul>
                                        </>
                                    )}

                                    {/* カスタムフィールドを動的に表示 */}
                                    {Object.keys(result)
                                        .filter(key => !['id', 'transcript', 'segments', 'summary', 'speakers', 'action_items', 'decisions'].includes(key))
                                        .sort()  // アルファベット順に表示
                                        .map(key => {
                                            const value = result[key];
                                            // null, undefined, 空配列, 空文字列はスキップ
                                            if (value === null || value === undefined) return null;
                                            if (Array.isArray(value) && value.length === 0) return null;
                                            if (typeof value === 'string' && value.trim() === '') return null;

                                            return (
                                                <div key={key} className="mt-4">
                                                    <h4>{key}</h4>
                                                    {Array.isArray(value) ? (
                                                        <ul className="summary-list">
                                                            {value.map((item, idx) => <li key={idx}>{item}</li>)}
                                                        </ul>
                                                    ) : (
                                                        <div className="summary-box mb-4">
                                                            {String(value)}
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                </div>
                            </div>
                        )}

                        {(state === 'processing' || state === 'summarizing') && (
                            <div className="flex flex-col items-center justify-center p-12 text-secondary">
                                <Loader2 size={48} className="animate-spin mb-4 text-primary" />
                                <p>{state === 'processing' ? text.processingAudio : text.generatingSummary}</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div >
    );
}

