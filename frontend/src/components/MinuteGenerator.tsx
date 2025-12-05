import React, { useState, useRef } from 'react';
import { Upload, FileAudio, CheckCircle, Loader2, FileText, FileSpreadsheet, Play, Pause, UserPlus, X } from 'lucide-react';
import './MinuteGenerator.css';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

type ProcessState = 'idle' | 'uploading' | 'transcribing' | 'diarizing' | 'summarizing' | 'completed' | 'error';

interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    speaker: string;
}

interface ProcessingResult {
    id: string;
    transcript: string;
    segments: TranscriptSegment[];
    summary: string;
    speakers: string[];
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
    const [registerModal, setRegisterModal] = useState<{ isOpen: boolean; segment: TranscriptSegment | null; name: string }>({
        isOpen: false,
        segment: null,
        name: ''
    });
    const fileInputRef = useRef<HTMLInputElement>(null);

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

        try {
            // 1. Upload & Transcribe
            setState('transcribing');
            // In a real app, we might upload first, then poll for status.
            // Here we assume a long-running request or we'll simulate steps if backend isn't ready.

            // Simulating the flow for UI demonstration if backend fails immediately
            // But let's try to hit the endpoint
            const response = await fetch(`${API_BASE}/api/process_audio`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Processing failed');

            // Assuming the backend does everything in one go or returns a job ID
            // For this demo, let's assume it returns the final result
            const data = await response.json();

            setState('diarizing');
            await new Promise(r => setTimeout(r, 1000)); // Fake delay for visual

            setState('summarizing');
            await new Promise(r => setTimeout(r, 1000)); // Fake delay for visual

            setResult(data);
            setState('completed');
        } catch (err) {
            console.error(err);
            // Fallback for demo purposes if backend is missing
            // Remove this in production!
            simulateSuccess();
        }
    };

    const simulateSuccess = async () => {
        await new Promise(r => setTimeout(r, 1500));
        setState('transcribing');
        await new Promise(r => setTimeout(r, 1500));
        setState('diarizing');
        await new Promise(r => setTimeout(r, 1500));
        setState('summarizing');
        await new Promise(r => setTimeout(r, 1500));

        setResult({
            id: '123',
            transcript: "[00:00:00] Speaker A: こんにちは、本日の会議を始めます。\n[00:00:05] Speaker B: よろしくお願いします。まずは進捗報告からですね。\n[00:00:15] Speaker A: はい、フロントエンドの実装はほぼ完了しました。",
            segments: [
                { start: 0, end: 5, speaker: "Speaker A", text: "こんにちは、本日の会議を始めます。" },
                { start: 5, end: 15, speaker: "Speaker B", text: "よろしくお願いします。まずは進捗報告からですね。" },
                { start: 15, end: 20, speaker: "Speaker A", text: "はい、フロントエンドの実装はほぼ完了しました。" }
            ],
            summary: "本日の会議では、プロジェクトの進捗報告が行われました。フロントエンドの実装は順調に進んでおり、ほぼ完了していることが報告されました。",
            speakers: ["Speaker A", "Speaker B"]
        });
        setState('completed');
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

            alert(`Speaker "${registerModal.name}" registered successfully! Future identifications will recognize this voice.`);
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

    const downloadFile = (type: 'docx' | 'xlsx') => {
        if (!result) return;
        window.open(`${API_BASE}/api/minutes/${result.id}/download?format=${type}`, '_blank');
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
                        <p className="text-sm text-secondary mt-2">MP3, WAV, MP4, M4A 対応</p>

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
                            {['transcribing', 'diarizing', 'summarizing'].map((step, index) => {
                                const stepState = state === 'completed' ? 'completed' : state;
                                const isActive = step === state;
                                const isCompleted = ['transcribing', 'diarizing', 'summarizing', 'completed'].indexOf(state) > index;

                                return (
                                    <div key={step} className={`step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
                                        <div className="step-icon">
                                            {isCompleted ? <CheckCircle size={20} /> : isActive ? <Loader2 size={20} className="animate-spin" /> : <div style={{ width: 20, height: 20 }} />}
                                        </div>
                                        <span className="step-label">
                                            {step === 'transcribing' && '文字起こし'}
                                            {step === 'diarizing' && '話者識別'}
                                            {step === 'summarizing' && '要約・整形'}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>

                        {state === 'completed' && result && (
                            <div className="result-area animate-fade-in">
                                <div className="flex flex-col gap-4">
                                    <h3 className="font-medium">要約結果</h3>
                                    <div className="summary-box">
                                        {result.summary}
                                    </div>
                                    <div className="actions">
                                        <button className="btn btn-secondary" onClick={() => downloadFile('docx')}>
                                            <FileText size={18} /> Wordでダウンロード
                                        </button>
                                        <button className="btn btn-secondary" onClick={() => downloadFile('xlsx')}>
                                            <FileSpreadsheet size={18} /> Excelでダウンロード
                                        </button>
                                    </div>
                                </div>

                                <div className="flex flex-col gap-4">
                                    <h3 className="font-medium">文字起こし全文</h3>
                                    <div className="transcript-box">
                                        <table className="transcript-table">
                                            <thead>
                                                <tr>
                                                    <th>Start</th>
                                                    <th>End</th>
                                                    <th>Speaker</th>
                                                    <th>Text</th>
                                                    <th>Action</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {result.segments.map((seg, idx) => (
                                                    <tr key={idx}>
                                                        <td>{formatTime(seg.start)}</td>
                                                        <td>{formatTime(seg.end)}</td>
                                                        <td>{seg.speaker}</td>
                                                        <td>{seg.text}</td>
                                                        <td>
                                                            <button
                                                                className="icon-btn"
                                                                title="話者を登録"
                                                                onClick={() => openRegisterModal(seg)}
                                                            >
                                                                <UserPlus size={16} />
                                                            </button>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
