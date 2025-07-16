class FeatureViewer {
    constructor() {
        this.data = [];
        this.currentIndex = 0;
        this.currentThreshold = 0;
        
        // DOM elements
        this.featureSelect = document.getElementById('feature-select');
        this.tokenTypeSelect = document.getElementById('token-type-select');
        this.activeTypeSelect = document.getElementById('active-type-select');
        this.thresholdSlider = document.getElementById('threshold-slider');
        this.thresholdValue = document.getElementById('threshold-value');
        this.loadBtn = document.getElementById('load-btn');
        this.status = document.getElementById('status');
        this.navigation = document.getElementById('navigation');
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.promptCounter = document.getElementById('prompt-counter');
        this.promptDisplay = document.getElementById('prompt-display');
        this.promptInfo = document.getElementById('prompt-info');
        this.promptText = document.getElementById('prompt-text');
        this.featureDescription = document.getElementById('feature-description');
        this.neuronpediaLink = document.getElementById('neuronpedia-link');
        
        this.initEventListeners();
        this.loadStoredSettings();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.thresholdSlider.addEventListener('input', (e) => this.updateThreshold(e.target.value));
        this.prevBtn.addEventListener('click', () => this.navigatePrompt(-1));
        this.nextBtn.addEventListener('click', () => this.navigatePrompt(1));
        
        // Save settings on change
        this.featureDescription.addEventListener('input', () => this.saveSettings());
        this.neuronpediaLink.addEventListener('input', () => this.saveSettings());
    }
    
    loadStoredSettings() {
        const settings = localStorage.getItem('featureViewerSettings');
        if (settings) {
            const parsed = JSON.parse(settings);
            this.featureDescription.value = parsed.description || '';
            this.neuronpediaLink.value = parsed.neuronpediaLink || '';
        }
    }
    
    saveSettings() {
        const settings = {
            description: this.featureDescription.value,
            neuronpediaLink: this.neuronpediaLink.value
        };
        localStorage.setItem('featureViewerSettings', JSON.stringify(settings));
    }
    
    updateThreshold(value) {
        this.currentThreshold = parseFloat(value);
        this.thresholdValue.textContent = value;
        if (this.data.length > 0) {
            this.displayPrompt();
        }
    }
    
    async loadData() {
        const feature = this.featureSelect.value;
        const tokenType = this.tokenTypeSelect.value;
        const activeType = this.activeTypeSelect.value;
        
        this.setStatus('Loading data...', 'loading');
        this.loadBtn.disabled = true;
        
        try {
            const fileName = this.constructFileName(activeType, tokenType);
            const filePath = `../gemma_trainer131k-l0-114_layer20/1000_prompts/${fileName}`;
            
            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const text = await response.text();
            this.data = text.trim().split('\\n').filter(line => line.trim()).map(line => JSON.parse(line));
            
            this.currentIndex = 0;
            this.setStatus(`Loaded ${this.data.length} prompts`, 'success');
            this.updateNavigation();
            this.displayPrompt();
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.setStatus(`Error loading data: ${error.message}`, 'error');
            this.data = [];
            this.hideNavigation();
        } finally {
            this.loadBtn.disabled = false;
        }
    }
    
    constructFileName(activeType, tokenType) {
        if (tokenType === 'all') {
            return `${activeType}.jsonl`;
        } else {
            return `${activeType}_${tokenType}.jsonl`;
        }
    }
    
    setStatus(message, type = '') {
        this.status.textContent = message;
        this.status.className = `status ${type}`;
    }
    
    updateNavigation() {
        if (this.data.length > 0) {
            this.navigation.style.display = 'flex';
            this.updateNavigationButtons();
        } else {
            this.hideNavigation();
        }
    }
    
    hideNavigation() {
        this.navigation.style.display = 'none';
    }
    
    updateNavigationButtons() {
        this.prevBtn.disabled = this.currentIndex === 0;
        this.nextBtn.disabled = this.currentIndex === this.data.length - 1;
        this.promptCounter.textContent = `${this.currentIndex + 1} / ${this.data.length}`;
    }
    
    navigatePrompt(direction) {
        const newIndex = this.currentIndex + direction;
        if (newIndex >= 0 && newIndex < this.data.length) {
            this.currentIndex = newIndex;
            this.updateNavigationButtons();
            this.displayPrompt();
        }
    }
    
    displayPrompt() {
        if (this.data.length === 0) {
            this.promptText.innerHTML = '<p>No data loaded</p>';
            this.promptInfo.style.display = 'none';
            return;
        }
        
        const prompt = this.data[this.currentIndex];
        this.displayPromptInfo(prompt);
        this.displayPromptText(prompt);
    }
    
    displayPromptInfo(prompt) {
        const feature = this.featureSelect.value;
        const maxActivation = prompt.max_feature_activations ? prompt.max_feature_activations[feature] : 0;
        
        let infoHTML = `
            <h3>Prompt ${prompt.prompt_id}</h3>
            <div class="prompt-info-grid">
                <div class="info-item">
                    <div class="info-label">Max Activation:</div>
                    <div class="info-value">${maxActivation.toFixed(4)}</div>
                </div>
        `;
        
        if (prompt.token_type) {
            infoHTML += `
                <div class="info-item">
                    <div class="info-label">Token Type:</div>
                    <div class="info-value">${prompt.token_type}</div>
                </div>
            `;
        }
        
        if (prompt.token_position !== undefined) {
            infoHTML += `
                <div class="info-item">
                    <div class="info-label">Token Position:</div>
                    <div class="info-value">${prompt.token_position}</div>
                </div>
            `;
        }
        
        if (prompt.max_activation_at_position !== undefined) {
            infoHTML += `
                <div class="info-item">
                    <div class="info-label">Activation at Position:</div>
                    <div class="info-value">${prompt.max_activation_at_position.toFixed(4)}</div>
                </div>
            `;
        }
        
        infoHTML += '</div>';
        
        this.promptInfo.innerHTML = infoHTML;
        this.promptInfo.style.display = 'block';
    }
    
    displayPromptText(prompt) {
        if (!prompt.tokenized_prompt) {
            this.promptText.innerHTML = `<p>${this.escapeHtml(prompt.prompt_text)}</p>`;
            return;
        }
        
        const feature = this.featureSelect.value;
        const tokenizedPrompt = prompt.tokenized_prompt;
        
        // Create activation map from tokens
        const activationMap = new Map();
        if (prompt.tokens) {
            prompt.tokens.forEach(token => {
                if (token.feature_activations && token.feature_activations[feature]) {
                    activationMap.set(token.position, token.feature_activations[feature]);
                }
            });
        }
        
        // Generate HTML with highlighting
        let html = '';
        for (let i = 0; i < tokenizedPrompt.length; i++) {
            const token = tokenizedPrompt[i];
            const activation = activationMap.get(i) || 0;
            
            if (activation > this.currentThreshold) {
                const colorClass = this.getActivationColorClass(activation);
                html += `<span class="token highlighted ${colorClass}">${this.escapeHtml(token)}</span>`;
            } else {
                html += `<span class="token">${this.escapeHtml(token)}</span>`;
            }
        }
        
        this.promptText.innerHTML = html;
    }
    
    getActivationColorClass(activation) {
        // Map activation to color class (0-10 scale)
        const level = Math.min(10, Math.max(0, Math.floor(activation)));
        return `activation-${level}`;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FeatureViewer();
});