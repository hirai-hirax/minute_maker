import React, { useState, useEffect } from 'react';
import { User, Trash2, Plus, Upload, Loader2, RefreshCw, Download, FileAudio } from 'lucide-react';
import './MinuteGenerator.css'; // Re-use styles for consistency

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

interface Speaker {
    name: string;
    path: string;
}

export function SpeakerManager() {
    const [speakers, setSpeakers] = useState<Speaker[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // New speaker form state
    const [newSpeakerName, setNewSpeakerName] = useState('');
    const [newSpeakerFile, setNewSpeakerFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);

    // Embedding generator state
    const [embedFile, setEmbedFile] = useState<File | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);

    const fetchSpeakers = async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/speakers`);
            if (!res.ok) throw new Error('Failed to fetch speakers');
            const data = await res.json();
            setSpeakers(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchSpeakers();
    }, []);

    const handleDelete = async (name: string) => {
        if (!confirm(`Are you sure you want to delete speaker "${name}"?`)) return;

        try {
            const res = await fetch(`${API_BASE}/api/speakers/${name}`, {
                method: 'DELETE',
            });
            if (!res.ok) throw new Error('Failed to delete speaker');

            // Refresh list
            fetchSpeakers();
        } catch (err: any) {
            alert(err.message);
        }
    };

    const handleAddSpeaker = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newSpeakerName || !newSpeakerFile) return;

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', newSpeakerFile);
        formData.append('name', newSpeakerName);

        try {
            const res = await fetch(`${API_BASE}/api/speakers`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.detail || 'Failed to add speaker');
            }

            alert(`Speaker "${newSpeakerName}" added successfully.`);
            setNewSpeakerName('');
            setNewSpeakerFile(null);
            fetchSpeakers();
        } catch (err: any) {
            alert(`Error: ${err.message}`);
            // Also log to console for debugging
            console.error(err);
        } finally {
            setIsUploading(false);
        }
    };

    const handleGenerateEmbedding = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!embedFile) return;

        setIsGenerating(true);
        const formData = new FormData();
        formData.append('file', embedFile);

        try {
            const res = await fetch(`${API_BASE}/api/create_speaker_embedding`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.detail || 'Failed to generate embedding');
            }

            // Trigger download
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;

            // Determine filename
            let filename = embedFile.name.replace(/\.[^/.]+$/, "") + ".npy";
            const disposition = res.headers.get('Content-Disposition');
            if (disposition && disposition.indexOf('attachment') !== -1) {
                const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                const matches = filenameRegex.exec(disposition);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }

            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();

            setEmbedFile(null);
        } catch (err: any) {
            alert(`Error: ${err.message}`);
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div className="minute-generator animate-fade-in">
            <div className="card">
                <div className="card__header mb-6">
                    <div>
                        <p className="eyebrow text-accent-primary">Management</p>
                        <h2>話者管理 (Speaker Manager)</h2>
                    </div>
                    <button className="btn btn-secondary" onClick={fetchSpeakers} disabled={isLoading}>
                        <RefreshCw size={16} className={isLoading ? "animate-spin" : ""} /> 更新
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* List Section */}
                    <div>
                        <h3 className="font-medium mb-4 flex items-center gap-2">
                            <User size={20} /> 登録済み話者一覧
                        </h3>

                        {isLoading ? (
                            <div className="text-center p-8 text-secondary">
                                <Loader2 size={32} className="animate-spin mx-auto mb-2" />
                                <p>読み込み中...</p>
                            </div>
                        ) : (
                            <div className="bg-secondary rounded-lg border border-border overflow-hidden">
                                {speakers.length === 0 ? (
                                    <div className="p-8 text-center text-secondary">
                                        登録された話者はいません。
                                    </div>
                                ) : (
                                    <ul className="divide-y divide-border">
                                        {speakers.map((spk) => (
                                            <li key={spk.name} className="p-4 flex justify-between items-center hover:bg-white hover:bg-opacity-5 transition-colors">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-accent-primary font-bold">
                                                        {spk.name.charAt(0).toUpperCase()}
                                                    </div>
                                                    <span className="font-medium">{spk.name}</span>
                                                </div>
                                                <button
                                                    className="text-secondary hover:text-red-400 p-2 transition-colors"
                                                    onClick={() => handleDelete(spk.name)}
                                                    title="削除"
                                                >
                                                    <Trash2 size={18} />
                                                </button>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Add Section */}
                    <div>
                        <h3 className="font-medium mb-4 flex items-center gap-2">
                            <Plus size={20} /> 新規登録
                        </h3>
                        <div className="bg-secondary p-6 rounded-lg border border-border">
                            <form onSubmit={handleAddSpeaker}>
                                <div className="mb-4">
                                    <label className="block text-sm text-secondary mb-2">話者名</label>
                                    <input
                                        type="text"
                                        className="input-field"
                                        value={newSpeakerName}
                                        onChange={(e) => setNewSpeakerName(e.target.value)}
                                        placeholder="例: 佐藤花子"
                                        required
                                    />
                                </div>
                                <div className="mb-6">
                                    <label className="block text-sm text-secondary mb-2">音声サンプル (WAV/MP3)</label>
                                    <div className="relative">
                                        <input
                                            type="file"
                                            className="hidden"
                                            id="speaker-upload"
                                            accept="audio/*"
                                            onChange={(e) => setNewSpeakerFile(e.target.files ? e.target.files[0] : null)}
                                        />
                                        <label
                                            htmlFor="speaker-upload"
                                            className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-accent-primary hover:bg-accent-primary hover:bg-opacity-5 transition-all"
                                        >
                                            <Upload size={24} className="mb-2 text-secondary" />
                                            <span className="text-sm text-secondary">
                                                {newSpeakerFile ? newSpeakerFile.name : "クリックして音声を選択"}
                                            </span>
                                        </label>
                                    </div>
                                </div>
                                <button
                                    type="submit"
                                    className="btn btn-primary w-full"
                                    disabled={!newSpeakerName || !newSpeakerFile || isUploading}
                                >
                                    {isUploading ? (
                                        <>
                                            <Loader2 size={18} className="animate-spin mr-2" /> 登録中...
                                        </>
                                    ) : (
                                        "登録する"
                                    )}
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                {/* Embedding Generator Section */}
                <div className="mt-8 pt-8 border-t border-border">
                    <h3 className="font-medium mb-4 flex items-center gap-2">
                        <FileAudio size={20} /> 埋め込みファイル生成ツール
                    </h3>
                    <div className="bg-secondary p-6 rounded-lg border border-border">
                        <div className="mb-4">
                            <p className="text-sm text-secondary mb-4">
                                音声ファイルをアップロードして、話者特徴量(Embedding)を抽出します。
                                結果は <code>.npy</code> ファイルとしてダウンロードされます。
                                この機能はサーバーに話者を登録せず、ファイル生成のみを行います。
                            </p>

                            <form onSubmit={handleGenerateEmbedding} className="flex gap-4 items-end">
                                <div className="flex-1">
                                    <div className="relative">
                                        <input
                                            type="file"
                                            className="hidden"
                                            id="embed-upload"
                                            accept="audio/*"
                                            onChange={(e) => setEmbedFile(e.target.files ? e.target.files[0] : null)}
                                        />
                                        <label
                                            htmlFor="embed-upload"
                                            className="flex flex-col items-center justify-center w-full h-24 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-accent-primary hover:bg-accent-primary hover:bg-opacity-5 transition-all"
                                        >
                                            {embedFile ? (
                                                <div className="text-center">
                                                    <span className="text-accent-primary font-medium block mb-1">
                                                        {embedFile.name}
                                                    </span>
                                                    <span className="text-xs text-secondary">クリックして変更</span>
                                                </div>
                                            ) : (
                                                <>
                                                    <Upload size={20} className="mb-2 text-secondary" />
                                                    <span className="text-sm text-secondary">
                                                        クリックして音声を選択
                                                    </span>
                                                </>
                                            )}
                                        </label>
                                    </div>
                                </div>
                                <button
                                    type="submit"
                                    className="btn btn-primary h-12 px-6 flex items-center gap-2"
                                    disabled={!embedFile || isGenerating}
                                >
                                    {isGenerating ? (
                                        <>
                                            <Loader2 size={18} className="animate-spin" /> 生成中...
                                        </>
                                    ) : (
                                        <>
                                            <Download size={18} /> 生成してダウンロード
                                        </>
                                    )}
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
