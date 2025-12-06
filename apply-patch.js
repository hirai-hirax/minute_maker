const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'frontend', 'src', 'components', 'MinuteGenerator.tsx');
let content = fs.readFileSync(filePath, 'utf8');

// Update handleSummarize to use FormData
const oldHandleSummarize = `    const handleSummarize = async () => {
        if (!result) return;
        setState('summarizing');

        const transcriptText = mergedTranscript.map(m => \`\${m.speaker}: \${m.text}\`).join("\\n");

        try {
            const response = await fetch(\`\${API_BASE}/api/generate_summary\`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transcript: transcriptText,
                    prompt_id: selectedPromptId
                }),
            });`;

const newHandleSummarize = `    const handleSummarize = async () => {
        if (!result) return;
        setState('summarizing');

        const transcriptText = mergedTranscript.map(m => \`\${m.speaker}: \${m.text}\`).join("\\n");

        try {
            // Use FormData to support file uploads
            const formData = new FormData();
            formData.append('transcript', transcriptText);
            formData.append('prompt_id', selectedPromptId);
            
            // Append reference files
            referenceFiles.forEach(file => {
                formData.append('reference_files', file);
            });

            const response = await fetch(\`\${API_BASE}/api/generate_summary\`, {
                method: 'POST',
                body: formData,
            });`;

content = content.replace(oldHandleSummarize, newHandleSummarize);

// Add UI section before "要約を実行" button
const uiSection = `
                                                    {/* Reference Materials Section */}
                                                    <div className="mt-6 pt-4 border-t border-border">
                                                        <h4 className="font-medium mb-2 flex items-center gap-2">
                                                            <Paperclip size={18} />
                                                            参考資料
                                                        </h4>
                                                        <p className="text-secondary text-xs mb-3">
                                                            要約に使用する参考資料をアップロード (任意)
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
                                                            ファイルを選択
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
                                                                            title="削除"
                                                                        >
                                                                            <X size={14} />
                                                                        </button>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
`;

// Find the button pattern and insert UI before it  
content = content.replace(
    /(\s+<button className="btn btn-primary[^>]*onClick=\{handleSummarize\}>)/,
    `${uiSection}$1`
);

fs.writeFileSync(filePath, content, 'utf8');
console.log('✅ MinuteGenerator.tsx patched successfully');
console.log('   - Updated handleSummarize to use FormData');
console.log('   - Added reference materials UI section');
