import React, { useState, useEffect } from 'react'
import { User, Trash2, Plus, Upload, Loader2, RefreshCw, Download, FileAudio } from 'lucide-react'
import './MinuteGenerator.css' // Re-use styles for consistency
import { useI18n } from '../i18n'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

const TEXT = {
    ja: {
        header: '話者管理 (Speaker Manager)',
        refresh: '更新',
        speakersSection: '1. 登録済み話者一覧',
        loading: '読み込み中...',
        empty: '登録された話者はいません。',
        deleteTitle: '削除',
        deleteConfirm: (name: string) => `話者「${name}」を削除してもよろしいですか？`,
        addSection: '2. 話者登録',
        speakerNameLabel: '話者名',
        speakerNamePlaceholder: '例: 佐藤花子',
        sampleLabel: '音声サンプル (WAV/MP3)',
        selectAudio: 'クリックして音声を選択',
        registering: '登録中...',
        register: '登録する',
        addSuccess: (name: string) => `話者「${name}」を登録しました。`,
        generateSection: '3. 埋め込みファイル生成ツール',
        generateDescription1: '音声ファイルをアップロードして、話者特徴量(Embedding)を抽出します。',
        generateDescription2: '結果は .npy ファイルとしてダウンロードされます。',
        generateDescription3: 'この機能はサーバーに話者を登録せず、ファイル生成のみを行います。',
        clickToChange: 'クリックして変更',
        generating: '生成中...',
        generateDownload: '生成してダウンロード',
        embedSelectAudio: 'クリックして音声を選択',
    },
    en: {
        header: 'Speaker Manager',
        refresh: 'Refresh',
        speakersSection: '1. Registered speakers',
        loading: 'Loading...',
        empty: 'No speakers registered.',
        deleteTitle: 'Delete',
        deleteConfirm: (name: string) => `Are you sure you want to delete speaker "${name}"?`,
        addSection: '2. Register speaker',
        speakerNameLabel: 'Speaker name',
        speakerNamePlaceholder: 'e.g., Hanako Sato',
        sampleLabel: 'Audio sample (WAV/MP3)',
        selectAudio: 'Click to choose audio',
        registering: 'Registering...',
        register: 'Register',
        addSuccess: (name: string) => `Speaker "${name}" added successfully.`,
        generateSection: '3. Embedding generator',
        generateDescription1: 'Upload audio to extract a speaker embedding.',
        generateDescription2: 'The result downloads as a .npy file.',
        generateDescription3: 'This feature only generates a file without registering the speaker.',
        clickToChange: 'Click to change',
        generating: 'Generating...',
        generateDownload: 'Generate & download',
        embedSelectAudio: 'Click to choose audio',
    }
} as const

interface Speaker {
    name: string;
    path: string;
}

export function SpeakerManager() {
    const { language } = useI18n()
    const text = TEXT[language]

    const [speakers, setSpeakers] = useState<Speaker[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // For adding speakers via audio
    const [newSpeakerName, setNewSpeakerName] = useState('');
    const [newSpeakerFile, setNewSpeakerFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);

    // For generating embeddings
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
        if (!confirm(text.deleteConfirm(name))) return;

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

            alert(text.addSuccess(newSpeakerName));
            setNewSpeakerName('');
            setNewSpeakerFile(null);
            fetchSpeakers();
        } catch (err: any) {
            alert(err.message);
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
                        <h2>{text.header}</h2>
                    </div>
                    <button className="btn btn-secondary" onClick={fetchSpeakers} disabled={isLoading}>
                        <RefreshCw size={16} className={isLoading ? "animate-spin" : ""} /> {text.refresh}
                    </button>
                </div>

                {/* ========== Section 1: Speaker List ========== */}
                <div className="mb-10 pb-8 border-b-4 border-border">
                    <div className="bg-gradient-to-r from-accent-primary to-transparent px-6 py-4 -mx-6 mb-6 rounded-lg">
                        <h3 className="text-xl font-bold text-white flex items-center gap-2">
                            <User size={24} /> {text.speakersSection}
                        </h3>
                    </div>

                    {isLoading ? (
                        <div className="text-center p-8 text-secondary">
                            <Loader2 size={32} className="animate-spin mx-auto mb-2" />
                            <p>{text.loading}</p>
                        </div>
                    ) : (
                        <div className="bg-secondary rounded-lg border border-border overflow-hidden">
                            {speakers.length === 0 ? (
                                <div className="p-8 text-center text-secondary">
                                    {text.empty}
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
                                                title={text.deleteTitle}
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

                {/* ========== Section 2: Register Speaker ========== */}
                <div className="mb-10 pb-8 border-b-4 border-border">
                    <div className="bg-gradient-to-r from-blue-600 to-transparent px-6 py-4 -mx-6 mb-6 rounded-lg">
                        <h3 className="text-xl font-bold text-white flex items-center gap-2">
                            <Plus size={24} /> {text.addSection}
                        </h3>
                    </div>

                    <div className="bg-secondary p-6 rounded-lg border border-border">
                        <form onSubmit={handleAddSpeaker}>
                            <div className="mb-4">
                                <label className="block text-sm text-secondary mb-2">{text.speakerNameLabel}</label>
                                <input
                                    type="text"
                                    className="input-field"
                                    value={newSpeakerName}
                                    onChange={(e) => setNewSpeakerName(e.target.value)}
                                    placeholder={text.speakerNamePlaceholder}
                                    required
                                />
                            </div>
                            <div className="mb-6">
                                <label className="block text-sm text-secondary mb-2">{text.sampleLabel}</label>
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
                                            {newSpeakerFile ? newSpeakerFile.name : text.selectAudio}
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
                                        <Loader2 size={18} className="animate-spin mr-2" /> {text.registering}
                                    </>
                                ) : (
                                    text.register
                                )}
                            </button>
                        </form>
                    </div>
                </div>

                {/* ========== Section 3: Embedding Generator ========== */}
                <div className="mb-8">
                    <div className="bg-gradient-to-r from-green-600 to-transparent px-6 py-4 -mx-6 mb-6 rounded-lg">
                        <h3 className="text-xl font-bold text-white flex items-center gap-2">
                            <FileAudio size={24} /> {text.generateSection}
                        </h3>
                    </div>

                    <div className="bg-secondary p-6 rounded-lg border border-border">
                        <p className="text-sm text-secondary mb-4">
                            {text.generateDescription1}
                            <br />
                            {text.generateDescription2}
                            <br />
                            {text.generateDescription3}
                        </p>

                        <form onSubmit={handleGenerateEmbedding} className="flex gap-4 items-end flex-col sm:flex-row">
                            <div className="flex-1 w-full">
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
                                                <span className="text-xs text-secondary">{text.clickToChange}</span>
                                            </div>
                                        ) : (
                                            <>
                                                <Upload size={20} className="mb-2 text-secondary" />
                                                <span className="text-sm text-secondary">
                                                    {text.embedSelectAudio}
                                                </span>
                                            </>
                                        )}
                                    </label>
                                </div>
                            </div>
                            <button
                                type="submit"
                                className="btn btn-primary h-12 px-6 flex items-center gap-2 w-full sm:w-auto"
                                disabled={!embedFile || isGenerating}
                            >
                                {isGenerating ? (
                                    <>
                                        <Loader2 size={18} className="animate-spin" /> {text.generating}
                                    </>
                                ) : (
                                    <>
                                        <FileAudio size={18} /> {text.generateDownload}
                                    </>
                                )}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
}
