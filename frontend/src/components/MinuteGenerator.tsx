import React, { useState, useRef } from 'react';
import { Upload, FileAudio, CheckCircle, Loader2, FileText, FileSpreadsheet, Play, Pause, UserPlus, X, RefreshCw, ArrowRight } from 'lucide-react';
import './MinuteGenerator.css';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

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
    summary: string;
    speakers: string[];
    action_items?: string[];
    decisions?: string[];
}

const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
};

export function MinuteGenerator() {
    const [state, setState] = useState<ProcessState>('idle');
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<ProcessingResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [prompts, setPrompts] = useState<PromptPreset[]>([]);
    const [selectedPromptId, setSelectedPromptId] = useState<string>('standard');
    const [selectedModel, setSelectedModel] = useState<'gpt-4o' | 'whisper'>('gpt-4o');
    const [mergedTranscript, setMergedTranscript] = useState<{ speaker: string; text: string }[]>([]);
    const [transcriptViewMode, setTranscriptViewMode] = useState<'structured' | 'text'>('structured');

    const [registerModal, setRegisterModal] = useState<{ isOpen: boolean; segment: TranscriptSegment | null; name: string }>({
        isOpen: false,
        segment: null,
        name: ''
    });
    const fileInputRef = useRef<HTMLInputElement>(null);

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

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', selectedModel);

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
            alert('Speaker identification complete!');
        } catch (err) {
            console.error(err);
            alert('Failed to identify speakers.');
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
            const speakerName = seg.speaker || "Unknown";

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
            .map((item) => `${item.speaker || 'Unknown'}: ${item.text}`)
            .join("\n\n");
    }, [mergedTranscript]);

    const handleSummarize = async () => {
        if (!result) return;
        setState('summarizing');

        const transcriptText = mergedTranscript.map(m => `${m.speaker}: ${m.text}`).join("\n");

        try {
            const response = await fetch(`${API_BASE}/api/generate_summary`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transcript: transcriptText,
                    prompt_id: selectedPromptId
                }),
            });

            if (!response.ok) throw new Error('Summarization failed');

            const summaryData = await response.json();
            setResult({
                ...result,
                summary: summaryData.summary,
                action_items: summaryData.action_items,
                decisions: summaryData.decisions
            });
            setState('completed');
        } catch (err) {
            console.error(err);
            alert('Summarization failed.');
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

            alert(`Speaker "${registerModal.name}" registered successfully! Click "Identify Speakers" to apply changes.`);
            setRegisterModal({ isOpen: false, segment: null, name: '' });
        } catch (e: any) {
            alert(`Error: ${e.message}`);
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
            const response = await fetch(`${API_BASE}/api/download_minutes?format=${type}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: `議事録_${new Date().toLocaleDateString('ja-JP').replace(/\//g, '-')}`,
                    summary: result.summary || '',
                    decisions: result.decisions || [],
                    action_items: result.action_items || [],
                    segments: result.segments
                }),
            });

            if (!response.ok) throw new Error('Download failed');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `議事録_${new Date().toISOString().split('T')[0]}.${type}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            console.error(err);
            alert(`ダウンロードに失敗しました: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
    };

    return (
        <div className="minute-generator animate-fade-in">
            {registerModal.isOpen && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <div className="modal-header">
                            <h3>話者登録</h3>
                            <button onClick={() => setRegisterModal({ ...registerModal, isOpen: false })}>
                                <X size={20} />
                            </button>
                        </div>
                        <div className="modal-body">
                            <p className="text-sm text-secondary mb-4">
                                以下のセグメントの音声を使用して、話者「{registerModal.name}」の特徴をシステムに登録します。
                            </p>
                            <div className="segment-preview">
                                "{registerModal.segment?.text}"
                            </div>
                            <label className="input-label mt-4">話者名</label>
                            <input
                                type="text"
                                className="input-field"
                                value={registerModal.name}
                                onChange={(e) => setRegisterModal({ ...registerModal, name: e.target.value })}
                                placeholder="例: 山田太郎"
                                autoFocus
                            />
                        </div>
                        <div className="modal-footer">
                            <button className="btn btn-secondary" onClick={() => setRegisterModal({ ...registerModal, isOpen: false })}>
                                キャンセル
                            </button>
                            <button className="btn btn-primary" onClick={handleRegisterSpeaker}>
                                登録する
                            </button>
                        </div>
                    </div>
                </div>
            )}
            <div className="card">
                <div className="card__header mb-4">
                    <div>
                        <p className="eyebrow text-accent-primary">AI 自動生成</p>
                        <h2>音声・動画から議事録を作成</h2>
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
                        <h3>{file ? file.name : "ファイルをドラッグ＆ドロップ"}</h3>
                        <p className="text-secondary mt-4">または クリックして選択</p>
                        <p className="text-sm text-secondary mt-2 mb-6">MP3, WAV, MP4, M4A 対応</p>

                        <div className="model-select-group bg-secondary p-4 rounded-lg border border-border inline-flex flex-col gap-2 mb-4 text-left">
                            <label className="text-xs text-secondary font-medium">使用モデルを選択</label>
                            <div className="flex gap-4">
                                <label className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="radio"
                                        name="model"
                                        value="gpt-4o"
                                        checked={selectedModel === 'gpt-4o'}
                                        onChange={() => setSelectedModel('gpt-4o')}
                                    />
                                    <span className="text-sm font-medium">High Quality (GPT-4o)</span>
                                </label>
                                <label className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="radio"
                                        name="model"
                                        value="whisper"
                                        checked={selectedModel === 'whisper'}
                                        onChange={() => setSelectedModel('whisper')}
                                    />
                                    <span className="text-sm font-medium">Standard (Whisper)</span>
                                </label>
                            </div>
                            <p className="text-xs text-secondary mt-1">
                                {selectedModel === 'gpt-4o'
                                    ? "高精度な文字起こしと話者識別が可能です。"
                                    : "より高速ですが、話者識別機能は制限されます。"}
                            </p>
                        </div>

                        {file && (
                            <button
                                className="btn btn-primary mt-8"
                                onClick={(e) => { e.stopPropagation(); startProcess(); }}
                            >
                                生成を開始する
                            </button>
                        )}
                    </div>
                )}

                {state !== 'idle' && state !== 'error' && (
                    <div className="process-view">
                        <div className="progress-steps">
                            {['processing', 'reviewing', 'summarizing_setup', 'completed'].map((step, idx) => {
                                const stepLabels: any = { processing: '文字起こし', reviewing: '確認・編集', summarizing_setup: '要約・整形', completed: '完了' };
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
                                        if (confirm('現在の作業内容は破棄されます。最初からやり直しますか？')) {
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
                                    <h3 className="font-medium">文字起こし結果</h3>
                                    <div className="flex gap-2">
                                        <button className="btn btn-secondary" onClick={handleIdentifySpeakers}>
                                            <RefreshCw size={16} /> 話者識別を実行
                                        </button>
                                        <button className="btn btn-primary" onClick={prepareSummarySetup}>
                                            要約・整形の確認へ <ArrowRight size={16} />
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
                                                            {seg.speaker || "Unknown"}
                                                        </div>
                                                    </td>
                                                    <td className="cell-text">{seg.text}</td>
                                                    <td className="cell-action">
                                                        <button
                                                            className="icon-btn"
                                                            title="この話者を登録"
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
                                        <p>AIが要約を作成中...</p>
                                    </div>
                                ) : (
                                    <>
                                        <h3 className="font-medium mb-4">要約生成の設定</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                            <div className="col-span-2">
                                                <div className="flex items-start justify-between gap-4 mb-2 flex-wrap">
                                                    <div>
                                                        <h4 className="text-sm font-medium text-secondary">整形済み文字起こし</h4>
                                                        <p className="text-xs text-secondary mt-1">確認用に複数行をまとめたテキスト形式を切り替えできます。</p>
                                                    </div>
                                                    <div className="view-toggle">
                                                        <button
                                                            className={`toggle-btn ${transcriptViewMode === 'structured' ? 'active' : ''}`}
                                                            onClick={() => setTranscriptViewMode('structured')}
                                                            type="button"
                                                        >
                                                            リスト表示
                                                        </button>
                                                        <button
                                                            className={`toggle-btn ${transcriptViewMode === 'text' ? 'active' : ''}`}
                                                            onClick={() => setTranscriptViewMode('text')}
                                                            type="button"
                                                        >
                                                            テキスト表示
                                                        </button>
                                                    </div>
                                                </div>
                                                {transcriptViewMode === 'structured' ? (
                                                    <div className="transcript-box" style={{ maxHeight: '400px' }}>
                                                        {mergedTranscript.map((item, idx) => (
                                                            <div key={idx} className="mb-4 pb-2 border-b border-gray-700 last:border-0">
                                                                <div className="font-bold text-accent-primary text-sm mb-1">{item.speaker || "Unknown"}</div>
                                                                <div className="text-sm leading-relaxed">{item.text}</div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <textarea
                                                        className="transcript-textarea"
                                                        value={formattedTranscriptText}
                                                        readOnly
                                                        aria-label="整形済み文字起こしのテキスト表示"
                                                    />
                                                )}
                                            </div>
                                            <div className="col-span-1">
                                                <div className="bg-secondary p-4 rounded-lg border border-border">
                                                    <h4 className="font-medium mb-4">プロンプト選択</h4>
                                                    <p className="text-secondary text-sm mb-4">
                                                        目的に合わせた要約スタイルを選択してください。
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

                                                    <button className="btn btn-primary w-full mt-6" onClick={handleSummarize}>
                                                        AI要約を実行
                                                    </button>

                                                    <button className="btn btn-secondary w-full mt-2" onClick={() => setState('reviewing')}>
                                                        戻る
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
                                    <h3 className="font-medium">要約結果</h3>
                                    <div className="actions">
                                        <button className="btn btn-secondary" onClick={() => downloadFile('docx')}>
                                            <FileText size={18} /> Word
                                        </button>
                                        <button className="btn btn-secondary" onClick={() => downloadFile('xlsx')}>
                                            <FileSpreadsheet size={18} /> Excel
                                        </button>
                                        <button className="btn btn-secondary" onClick={() => setState('reviewing')}>
                                            戻る
                                        </button>
                                    </div>
                                </div>

                                <div className="summary-section animate-fade-in">
                                    <h4>概要</h4>
                                    <div className="summary-box mb-4">
                                        {result.summary}
                                    </div>

                                    {(result.decisions && result.decisions.length > 0) && (
                                        <>
                                            <h4>決定事項</h4>
                                            <ul className="summary-list">
                                                {result.decisions.map((d, i) => <li key={i}>{d}</li>)}
                                            </ul>
                                        </>
                                    )}

                                    {(result.action_items && result.action_items.length > 0) && (
                                        <>
                                            <h4>アクションアイテム</h4>
                                            <ul className="summary-list">
                                                {result.action_items.map((i, idx) => <li key={idx}>{i}</li>)}
                                            </ul>
                                        </>
                                    )}
                                </div>
                            </div>
                        )}

                        {(state === 'processing' || state === 'summarizing') && (
                            <div className="flex flex-col items-center justify-center p-12 text-secondary">
                                <Loader2 size={48} className="animate-spin mb-4 text-primary" />
                                <p>{state === 'processing' ? '音声処理中...' : 'AIが要約を作成中...'}</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div >
    );
}

