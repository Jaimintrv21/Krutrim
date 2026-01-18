/**
 * RLG Engine Frontend Application
 * Handles chat interface, document management, and analytics
 */

class RLGApp {
    constructor() {
        this.apiUrl = localStorage.getItem('rlg-api-url') || 'http://localhost:8000';
        this.settings = {
            topK: parseInt(localStorage.getItem('rlg-topK')) || 5,
            minReliability: parseFloat(localStorage.getItem('rlg-minReliability')) || 0.5,
            showConfidence: localStorage.getItem('rlg-showConfidence') !== 'false',
            darkMode: localStorage.getItem('rlg-darkMode') !== 'false'
        };
        this.currentSection = 'chat';
        this.selectedFile = null;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.checkConnection();
        this.loadDocuments();
        this.loadAnalytics();
        this.initSettings();
        
        // Auto-resize textarea
        const textarea = document.getElementById('questionInput');
        textarea.addEventListener('input', () => this.autoResize(textarea));
    }
    
    bindEvents() {
        // Sidebar navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.dataset.section;
                this.switchSection(section);
            });
        });
        
        // Sidebar toggle
        document.getElementById('sidebarToggle').addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('collapsed');
        });
        
        // Chat
        document.getElementById('sendBtn').addEventListener('click', () => this.sendQuestion());
        document.getElementById('questionInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendQuestion();
            }
        });
        document.getElementById('clearChatBtn').addEventListener('click', () => this.clearChat());
        
        // Document upload
        document.getElementById('uploadBtn').addEventListener('click', () => this.openUploadModal());
        document.getElementById('closeUploadModal').addEventListener('click', () => this.closeUploadModal());
        document.getElementById('cancelUpload').addEventListener('click', () => this.closeUploadModal());
        document.getElementById('confirmUpload').addEventListener('click', () => this.uploadDocument());
        
        // Upload zone interactions
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        
        // Reliability slider
        document.getElementById('docReliability').addEventListener('input', (e) => {
            document.getElementById('reliabilityValue').textContent = Math.round(e.target.value * 100) + '%';
        });
        
        // Settings
        document.getElementById('testConnectionBtn').addEventListener('click', () => this.testConnection());
        document.getElementById('apiUrl').addEventListener('change', (e) => {
            this.apiUrl = e.target.value;
            localStorage.setItem('rlg-api-url', this.apiUrl);
        });
        document.getElementById('topK').addEventListener('input', (e) => {
            this.settings.topK = parseInt(e.target.value);
            document.getElementById('topKValue').textContent = e.target.value;
            localStorage.setItem('rlg-topK', e.target.value);
        });
        document.getElementById('minReliability').addEventListener('input', (e) => {
            this.settings.minReliability = parseFloat(e.target.value);
            document.getElementById('minReliabilityValue').textContent = Math.round(e.target.value * 100) + '%';
            localStorage.setItem('rlg-minReliability', e.target.value);
        });
        document.getElementById('showConfidence').addEventListener('change', (e) => {
            this.settings.showConfidence = e.target.checked;
            localStorage.setItem('rlg-showConfidence', e.target.checked);
        });
        
        // Modal overlay click to close
        document.querySelector('.modal-overlay').addEventListener('click', () => this.closeUploadModal());
    }
    
    switchSection(section) {
        // Update nav
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.section === section);
        });
        
        // Update content
        document.querySelectorAll('.content-section').forEach(sec => {
            sec.classList.remove('active');
        });
        document.getElementById(`${section}Section`).classList.add('active');
        
        this.currentSection = section;
    }
    
    autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
    
    // =====================================
    // Connection & Status
    // =====================================
    
    async checkConnection() {
        const indicator = document.getElementById('statusIndicator');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('span');
        
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            if (response.ok) {
                dot.className = 'status-dot online';
                text.textContent = 'Connected';
            } else {
                throw new Error('Not OK');
            }
        } catch (e) {
            dot.className = 'status-dot offline';
            text.textContent = 'Disconnected';
        }
    }
    
    async testConnection() {
        const btn = document.getElementById('testConnectionBtn');
        btn.disabled = true;
        btn.textContent = 'Testing...';
        
        try {
            const response = await fetch(`${this.apiUrl}/`);
            const data = await response.json();
            
            if (response.ok) {
                alert(`✓ Connected to ${data.name} v${data.version}\n\nLLM Available: ${data.llm_available ? 'Yes' : 'No'}\nModel: ${data.llm_model}\nVectors: ${data.vector_index?.total_vectors || 0}`);
                this.checkConnection();
            } else {
                throw new Error('Connection failed');
            }
        } catch (e) {
            alert(`✗ Connection failed: ${e.message}\n\nMake sure the backend is running:\nuvicorn app.main:app --port 8000`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Test Connection';
        }
    }
    
    // =====================================
    // Chat Functions
    // =====================================
    
    async sendQuestion() {
        const input = document.getElementById('questionInput');
        const question = input.value.trim();
        
        if (!question) return;
        
        const requireGrounding = document.getElementById('requireGrounding').checked;
        const extractiveMode = document.getElementById('extractiveMode').checked;
        
        // Add user message
        this.addMessage(question, 'user');
        input.value = '';
        this.autoResize(input);
        
        // Add loading indicator
        const loadingId = this.addLoadingMessage();
        
        try {
            const endpoint = extractiveMode ? '/query/extractive' : '/query/';
            const response = await fetch(`${this.apiUrl}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    require_grounding: requireGrounding,
                    top_k: this.settings.topK,
                    min_reliability: this.settings.minReliability,
                    include_sources: true
                })
            });
            
            const data = await response.json();
            
            // Remove loading
            this.removeMessage(loadingId);
            
            if (response.ok) {
                if (data.status === 'no_grounded_answer') {
                    this.addMessage(this.formatNoAnswer(data), 'assistant', null, data);
                } else {
                    this.addMessage(data.answer, 'assistant', data.grounding_score, data);
                }
            } else {
                this.addMessage(`Error: ${data.detail || 'Failed to get response'}`, 'assistant');
            }
        } catch (e) {
            this.removeMessage(loadingId);
            this.addMessage(`Connection error: ${e.message}. Make sure the backend is running.`, 'assistant');
        }
        
        // Refresh analytics
        this.loadAnalytics();
    }
    
    addMessage(content, type, groundingScore = null, fullData = null) {
        const container = document.getElementById('chatMessages');
        const messageId = `msg-${Date.now()}`;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.id = messageId;
        
        let avatarSvg = '';
        if (type === 'user') {
            avatarSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
        } else {
            avatarSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>';
        }
        
        let scoreHtml = '';
        if (groundingScore !== null && this.settings.showConfidence) {
            const scoreClass = groundingScore >= 0.7 ? 'high' : groundingScore >= 0.5 ? 'medium' : 'low';
            scoreHtml = `
                <div class="grounding-score ${scoreClass}">
                    <span>Grounding: ${Math.round(groundingScore * 100)}%</span>
                </div>
            `;
        }
        
        let citationsHtml = '';
        if (fullData && fullData.sources_used && fullData.sources_used.length > 0) {
            const citationItems = fullData.sources_used.slice(0, 5).map((source, i) => `
                <div class="citation-item">
                    <span class="citation-marker">${i + 1}</span>
                    <div class="citation-text">
                        <div class="citation-source">${source.document_name}${source.page_number ? ` • p.${source.page_number}` : ''}</div>
                        ${source.excerpt ? `<div style="margin-top: 4px; font-size: 0.8rem; opacity: 0.8;">"${source.excerpt.substring(0, 100)}..."</div>` : ''}
                    </div>
                </div>
            `).join('');
            
            citationsHtml = `
                <div class="citations">
                    <div class="citations-title">Sources</div>
                    ${citationItems}
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatarSvg}</div>
            <div class="message-content">
                <p>${this.formatContent(content)}</p>
                ${scoreHtml}
                ${citationsHtml}
            </div>
        `;
        
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
        
        return messageId;
    }
    
    addLoadingMessage() {
        const container = document.getElementById('chatMessages');
        const messageId = `loading-${Date.now()}`;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.id = messageId;
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
        
        return messageId;
    }
    
    removeMessage(messageId) {
        const message = document.getElementById(messageId);
        if (message) {
            message.remove();
        }
    }
    
    formatContent(content) {
        // Convert markdown-style elements
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\[(\d+)\]/g, '<span class="citation-marker">$1</span>')
            .replace(/\n/g, '<br>');
    }
    
    formatNoAnswer(data) {
        let response = `<strong>Unable to provide a grounded answer.</strong><br><br>`;
        response += `<em>Reason:</em> ${data.reason}<br><br>`;
        
        if (data.suggestions && data.suggestions.length > 0) {
            response += `<em>Suggestions:</em><br>`;
            data.suggestions.forEach(s => {
                response += `• ${s}<br>`;
            });
        }
        
        if (data.partial_info) {
            response += `<br><em>Partial information found:</em><br>${data.partial_info}`;
        }
        
        response += `<br><br><small>Sources checked: ${data.sources_checked}</small>`;
        
        return response;
    }
    
    clearChat() {
        const container = document.getElementById('chatMessages');
        container.innerHTML = `
            <div class="message system-message">
                <div class="message-content">
                    <div class="welcome-card">
                        <div class="welcome-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                                <path d="M2 17l10 5 10-5"/>
                                <path d="M2 12l10 5 10-5"/>
                            </svg>
                        </div>
                        <h2>Welcome to RLG Engine</h2>
                        <p>Ask questions about your uploaded documents. Every answer comes with verified citations and a grounding score.</p>
                        <div class="feature-pills">
                            <span class="pill">🎯 Near-Zero Hallucination</span>
                            <span class="pill">📄 Source Citations</span>
                            <span class="pill">🔒 100% Offline</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // =====================================
    // Document Functions
    // =====================================
    
    async loadDocuments() {
        try {
            const response = await fetch(`${this.apiUrl}/documents/`);
            if (response.ok) {
                const data = await response.json();
                this.renderDocuments(data.documents);
            }
        } catch (e) {
            console.log('Could not load documents');
        }
    }
    
    renderDocuments(documents) {
        const grid = document.getElementById('documentsGrid');
        const empty = document.getElementById('emptyDocuments');
        
        if (!documents || documents.length === 0) {
            grid.innerHTML = '';
            grid.appendChild(empty);
            return;
        }
        
        grid.innerHTML = documents.map(doc => this.createDocumentCard(doc)).join('');
    }
    
    createDocumentCard(doc) {
        const iconClass = {
            'pdf': 'pdf',
            'docx': 'docx',
            'txt': 'txt',
            'html': 'html',
            'md': 'txt'
        }[doc.file_type] || 'txt';
        
        const statusColor = doc.status === 'indexed' ? 'var(--accent-success)' : 
                           doc.status === 'processing' ? 'var(--accent-warning)' : 
                           'var(--accent-danger)';
        
        return `
            <div class="document-card" data-id="${doc.id}">
                <div class="document-header">
                    <div class="document-icon ${iconClass}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <path d="M14 2v6h6"/>
                        </svg>
                    </div>
                    <div class="document-info">
                        <div class="document-name" title="${doc.filename}">${doc.filename}</div>
                        <div class="document-meta">
                            <span style="color: ${statusColor}">● ${doc.status}</span>
                            <span>${doc.file_type.toUpperCase()}</span>
                        </div>
                    </div>
                </div>
                <div class="document-stats">
                    <div class="stat">
                        <span class="stat-label">Pages</span>
                        <span class="stat-value">${doc.page_count || 0}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Chunks</span>
                        <span class="stat-value">${doc.chunk_count || 0}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Reliability</span>
                        <span class="stat-value">${Math.round(doc.reliability_score * 100)}%</span>
                    </div>
                </div>
                <div class="document-actions">
                    <button class="btn btn-ghost" onclick="app.deleteDocument('${doc.id}')">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                            <path d="M3 6h18"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>
                        </svg>
                        Delete
                    </button>
                </div>
            </div>
        `;
    }
    
    openUploadModal() {
        document.getElementById('uploadModal').classList.add('active');
        this.resetUploadForm();
    }
    
    closeUploadModal() {
        document.getElementById('uploadModal').classList.remove('active');
        this.resetUploadForm();
    }
    
    resetUploadForm() {
        document.getElementById('uploadZone').style.display = 'block';
        document.getElementById('uploadForm').style.display = 'none';
        document.getElementById('fileInput').value = '';
        document.getElementById('docTitle').value = '';
        document.getElementById('docCategory').value = '';
        document.getElementById('docReliability').value = 1;
        document.getElementById('reliabilityValue').textContent = '100%';
        document.getElementById('confirmUpload').disabled = true;
        this.selectedFile = null;
    }
    
    handleFileSelect(file) {
        this.selectedFile = file;
        
        document.getElementById('uploadZone').style.display = 'none';
        document.getElementById('uploadForm').style.display = 'block';
        document.getElementById('selectedFile').innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <path d="M14 2v6h6"/>
            </svg>
            <div>
                <strong>${file.name}</strong>
                <div style="font-size: 0.8rem; color: var(--text-tertiary);">
                    ${this.formatFileSize(file.size)}
                </div>
            </div>
        `;
        document.getElementById('confirmUpload').disabled = false;
    }
    
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
    
    async uploadDocument() {
        if (!this.selectedFile) return;
        
        const btn = document.getElementById('confirmUpload');
        btn.disabled = true;
        btn.textContent = 'Uploading...';
        
        const formData = new FormData();
        formData.append('file', this.selectedFile);
        formData.append('title', document.getElementById('docTitle').value);
        formData.append('category', document.getElementById('docCategory').value);
        formData.append('reliability_score', document.getElementById('docReliability').value);
        
        try {
            const response = await fetch(`${this.apiUrl}/documents/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                this.closeUploadModal();
                this.loadDocuments();
                this.loadAnalytics();
                
                // Show success message
                this.addMessage(`Document "${data.filename}" uploaded successfully! It has ${data.chunk_count} chunks ready for querying.`, 'assistant');
                this.switchSection('chat');
            } else {
                const error = await response.json();
                alert(`Upload failed: ${error.detail || 'Unknown error'}`);
            }
        } catch (e) {
            alert(`Upload failed: ${e.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Upload';
        }
    }
    
    async deleteDocument(docId) {
        if (!confirm('Are you sure you want to delete this document?')) return;
        
        try {
            const response = await fetch(`${this.apiUrl}/documents/${docId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.loadDocuments();
                this.loadAnalytics();
            } else {
                alert('Failed to delete document');
            }
        } catch (e) {
            alert(`Delete failed: ${e.message}`);
        }
    }
    
    // =====================================
    // Analytics Functions
    // =====================================
    
    async loadAnalytics() {
        try {
            // Load query stats
            const statsResponse = await fetch(`${this.apiUrl}/query/stats`);
            if (statsResponse.ok) {
                const stats = await statsResponse.json();
                document.getElementById('totalQueries').textContent = stats.total_queries || 0;
                document.getElementById('groundingRate').textContent = Math.round((stats.grounding_rate || 0) * 100) + '%';
                document.getElementById('avgGrounding').textContent = Math.round((stats.average_grounding_score || 0) * 100) + '%';
            }
            
            // Load document count
            const docsResponse = await fetch(`${this.apiUrl}/documents/`);
            if (docsResponse.ok) {
                const docs = await docsResponse.json();
                document.getElementById('totalDocuments').textContent = docs.total || 0;
            }
        } catch (e) {
            console.log('Could not load analytics');
        }
    }
    
    // =====================================
    // Settings Functions
    // =====================================
    
    initSettings() {
        document.getElementById('apiUrl').value = this.apiUrl;
        document.getElementById('topK').value = this.settings.topK;
        document.getElementById('topKValue').textContent = this.settings.topK;
        document.getElementById('minReliability').value = this.settings.minReliability;
        document.getElementById('minReliabilityValue').textContent = Math.round(this.settings.minReliability * 100) + '%';
        document.getElementById('showConfidence').checked = this.settings.showConfidence;
        document.getElementById('darkMode').checked = this.settings.darkMode;
    }
}

// Initialize app
const app = new RLGApp();
