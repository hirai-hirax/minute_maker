import React, { useState, useEffect } from 'react'
import { Save, Plus, Edit2, Trash2, X, CheckCircle2, FileText } from 'lucide-react'
import './PromptManager.css'
import { useI18n } from '../i18n'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

const TEXT = {
    ja: {
        loadFail: 'プロンプトの読み込みに失敗しました',
        missingFields: 'すべてのフィールドを入力してください',
        createFail: 'プロンプトの作成に失敗しました',
        updateFail: 'プロンプトの更新に失敗しました',
        deleteConfirm: 'このプロンプトを削除してもよろしいですか？',
        deleteFail: 'プロンプトの削除に失敗しました',
        defaultEditBlock: 'デフォルトプロンプトは編集できません',
        loading: '読み込み中...',
        title: 'プロンプト管理',
        subtitle: '要約生成用のプロンプトを管理します',
        newPrompt: '新規プロンプト作成',
        editPrompt: 'プロンプト編集',
        promptName: 'プロンプト名',
        namePlaceholder: '例: 簡潔な要約',
        description: '説明',
        descriptionPlaceholder: '例: 簡潔で分かりやすい要約を生成',
        systemPrompt: 'システムプロンプト',
        systemPlaceholder: 'あなたは議事録作成の専門家です。...',
        systemHint: 'LLMに送信される指示文です。要約の形式や出力フォーマットを指定してください。',
        cancel: 'キャンセル',
        update: '更新',
        create: '作成',
        defaultBadge: 'デフォルト',
        systemPromptLabel: 'システムプロンプト:',
        edit: '編集',
        delete: '削除',
        saved: '保存しました',
    },
    en: {
        loadFail: 'Failed to load prompts',
        missingFields: 'Please complete all fields.',
        createFail: 'Failed to create prompt',
        updateFail: 'Failed to update prompt',
        deleteConfirm: 'Are you sure you want to delete this prompt?',
        deleteFail: 'Failed to delete prompt',
        defaultEditBlock: 'Default prompts cannot be edited',
        loading: 'Loading...',
        title: 'Prompt Manager',
        subtitle: 'Manage prompts used for summarization',
        newPrompt: 'Create prompt',
        editPrompt: 'Edit prompt',
        promptName: 'Prompt name',
        namePlaceholder: 'e.g., Concise summary',
        description: 'Description',
        descriptionPlaceholder: 'e.g., Generate a concise and clear summary',
        systemPrompt: 'System prompt',
        systemPlaceholder: 'You are a meeting-minutes expert...',
        systemHint: 'Instruction sent to the LLM. Specify the summary style and output format.',
        cancel: 'Cancel',
        update: 'Update',
        create: 'Create',
        defaultBadge: 'Default',
        systemPromptLabel: 'System prompt:',
        edit: 'Edit',
        delete: 'Delete',
        saved: 'Saved',
    }
} as const

interface Prompt {
    id: string;
    name: string;
    description: string;
    system_prompt: string;
    is_default: boolean;
}

export function PromptManager() {
    const { language } = useI18n()
    const text = TEXT[language]

    const [prompts, setPrompts] = useState<Prompt[]>([]);
    const [loading, setLoading] = useState(true);
    const [saveSuccess, setSaveSuccess] = useState(false);
    const [editingPrompt, setEditingPrompt] = useState<Prompt | null>(null);
    const [isCreating, setIsCreating] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        description: '',
        system_prompt: ''
    });

    useEffect(() => {
        loadPrompts();
    }, []);

    const loadPrompts = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/prompts`);
            if (!response.ok) throw new Error('Failed to load prompts');
            const data = await response.json();
            setPrompts(data);
        } catch (error) {
            console.error('Failed to load prompts:', error);
            alert(text.loadFail);
        } finally {
            setLoading(false);
        }
    };

    const handleCreate = async () => {
        if (!formData.name.trim() || !formData.description.trim() || !formData.system_prompt.trim()) {
            alert(text.missingFields);
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/api/prompts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to create prompt');
            }

            setSaveSuccess(true);
            setTimeout(() => setSaveSuccess(false), 3000);
            setFormData({ name: '', description: '', system_prompt: '' });
            setIsCreating(false);
            await loadPrompts();
        } catch (error) {
            console.error('Failed to create prompt:', error);
            alert(`${text.createFail}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    const handleUpdate = async () => {
        if (!editingPrompt) return;

        try {
            const response = await fetch(`${API_BASE}/api/prompts/${editingPrompt.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: formData.name,
                    description: formData.description,
                    system_prompt: formData.system_prompt
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to update prompt');
            }

            setSaveSuccess(true);
            setTimeout(() => setSaveSuccess(false), 3000);
            setEditingPrompt(null);
            setFormData({ name: '', description: '', system_prompt: '' });
            await loadPrompts();
        } catch (error) {
            console.error('Failed to update prompt:', error);
            alert(`${text.updateFail}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    const handleDelete = async (promptId: string) => {
        if (!confirm(text.deleteConfirm)) return;

        try {
            const response = await fetch(`${API_BASE}/api/prompts/${promptId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete prompt');
            }

            setSaveSuccess(true);
            setTimeout(() => setSaveSuccess(false), 3000);
            await loadPrompts();
        } catch (error) {
            console.error('Failed to delete prompt:', error);
            alert(`${text.deleteFail}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    const startEdit = (prompt: Prompt) => {
        if (prompt.is_default) {
            alert(text.defaultEditBlock);
            return;
        }
        setEditingPrompt(prompt);
        setFormData({
            name: prompt.name,
            description: prompt.description,
            system_prompt: prompt.system_prompt
        });
        setIsCreating(false);
    };

    const cancelEdit = () => {
        setEditingPrompt(null);
        setIsCreating(false);
        setFormData({ name: '', description: '', system_prompt: '' });
    };

    const startCreate = () => {
        setIsCreating(true);
        setEditingPrompt(null);
        setFormData({ name: '', description: '', system_prompt: '' });
    };

    if (loading) {
        return (
            <div className="prompt-manager-container">
                <div className="loading">{text.loading}</div>
            </div>
        );
    }

    return (
        <div className="prompt-manager-container">
            <div className="prompt-manager-header">
                <div>
                    <h2>{text.title}</h2>
                    <p className="prompt-manager-subtitle">{text.subtitle}</p>
                </div>
                {!isCreating && !editingPrompt && (
                    <button className="btn-primary" onClick={startCreate}>
                        <Plus size={16} />
                        {text.newPrompt}
                    </button>
                )}
            </div>

            {/* Create/Edit Form */}
            {(isCreating || editingPrompt) && (
                <div className="prompt-form-card">
                    <div className="prompt-form-header">
                        <h3>{editingPrompt ? text.editPrompt : text.newPrompt}</h3>
                        <button className="icon-btn" onClick={cancelEdit}>
                            <X size={20} />
                        </button>
                    </div>
                    <div className="prompt-form-body">
                        <div className="input-group">
                            <label htmlFor="prompt-name">{text.promptName}</label>
                            <input
                                type="text"
                                id="prompt-name"
                                value={formData.name}
                                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                placeholder={text.namePlaceholder}
                                maxLength={100}
                            />
                        </div>
                        <div className="input-group">
                            <label htmlFor="prompt-description">{text.description}</label>
                            <input
                                type="text"
                                id="prompt-description"
                                value={formData.description}
                                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                placeholder={text.descriptionPlaceholder}
                                maxLength={200}
                            />
                        </div>
                        <div className="input-group">
                            <label htmlFor="prompt-system">{text.systemPrompt}</label>
                            <textarea
                                id="prompt-system"
                                value={formData.system_prompt}
                                onChange={(e) => setFormData({ ...formData, system_prompt: e.target.value })}
                                placeholder={text.systemPlaceholder}
                                rows={8}
                            />
                            <small className="help-text">
                                {text.systemHint}
                            </small>
                        </div>
                        <div className="form-actions">
                            <button className="btn-secondary" onClick={cancelEdit}>
                                {text.cancel}
                            </button>
                            <button
                                className="btn-primary"
                                onClick={editingPrompt ? handleUpdate : handleCreate}
                            >
                                <Save size={16} />
                                {editingPrompt ? text.update : text.create}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Prompts List */}
            <div className="prompts-grid">
                {prompts.map((prompt) => (
                    <div key={prompt.id} className="prompt-card">
                        <div className="prompt-card-header">
                            <div className="prompt-card-title">
                                <FileText size={20} className="prompt-icon" />
                                <h3>{prompt.name}</h3>
                            </div>
                            {prompt.is_default && (
                                <span className="badge-default">{text.defaultBadge}</span>
                            )}
                        </div>
                        <p className="prompt-card-description">{prompt.description}</p>
                        <div className="prompt-card-preview">
                            <strong>{text.systemPromptLabel}</strong>
                            <div className="prompt-preview-text">
                                {prompt.system_prompt.substring(0, 150)}
                                {prompt.system_prompt.length > 150 && '...'}
                            </div>
                        </div>
                        {!prompt.is_default && (
                            <div className="prompt-card-actions">
                                <button
                                    className="btn-secondary btn-sm"
                                    onClick={() => startEdit(prompt)}
                                >
                                    <Edit2 size={14} />
                                    {text.edit}
                                </button>
                                <button
                                    className="btn-danger btn-sm"
                                    onClick={() => handleDelete(prompt.id)}
                                >
                                    <Trash2 size={14} />
                                    {text.delete}
                                </button>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Success Message */}
            {saveSuccess && (
                <div className="save-success">
                    <CheckCircle2 size={20} />
                    {text.saved}
                </div>
            )}
        </div>
    );
}
