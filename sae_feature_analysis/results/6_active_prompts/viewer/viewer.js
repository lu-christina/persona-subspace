class FeatureViewer {
    constructor() {
        this.data = [];
        this.currentPage = 0;
        this.promptsPerPage = 50;
        this.currentThreshold = 0;
        this.features = [];
        this.featureMetadata = new Map();
        
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
        this.initializeViewer();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.thresholdSlider.addEventListener('input', (e) => this.updateThreshold(e.target.value));
        this.prevBtn.addEventListener('click', () => this.navigatePage(-1));
        this.nextBtn.addEventListener('click', () => this.navigatePage(1));
        this.featureSelect.addEventListener('change', () => this.updateFeatureMetadata());
    }
    
    updateThreshold(value) {
        this.currentThreshold = parseFloat(value);
        this.thresholdValue.textContent = value;
        if (this.data.length > 0) {
            this.displayPage();
        }
    }
    
    async loadData() {
        const feature = this.featureSelect.value;
        const tokenType = this.tokenTypeSelect.value;
        const activeType = this.activeTypeSelect.value;
        
        if (!feature) {
            this.setStatus('Please select a feature', 'error');
            return;
        }
        
        this.setStatus('Loading data...', 'loading');
        this.loadBtn.disabled = true;
        
        try {
            const fileName = this.constructFileName(activeType, tokenType);
            const filePath = `../gemma_trainer131k-l0-114_layer20/1000_prompts/${feature}/${fileName}`;
            
            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const text = await response.text();
            
            // Parse JSONL more carefully to handle malformed JSON
            this.data = [];
            const lines = text.trim().split('\n');
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line) {
                    try {
                        const parsed = JSON.parse(line);
                        this.data.push(parsed);
                    } catch (jsonError) {
                        console.warn(`Skipping malformed JSON on line ${i + 1}:`, jsonError.message);
                        console.warn(`Line content:`, line);
                    }
                }
            }
            
            this.currentPage = 0;
            this.setStatus(`Loaded ${this.data.length} prompts`, 'success');
            this.updateNavigation();
            this.displayPage();
            
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
            this.navigation.style.display = 'block';
            this.updateNavigationButtons();
        } else {
            this.hideNavigation();
        }
    }
    
    hideNavigation() {
        this.navigation.style.display = 'none';
    }
    
    updateNavigationButtons() {
        const totalPages = Math.ceil(this.data.length / this.promptsPerPage);
        this.prevBtn.disabled = this.currentPage === 0;
        this.nextBtn.disabled = this.currentPage === totalPages - 1;
        
        const startPrompt = this.currentPage * this.promptsPerPage + 1;
        const endPrompt = Math.min((this.currentPage + 1) * this.promptsPerPage, this.data.length);
        this.promptCounter.textContent = `${startPrompt}-${endPrompt} of ${this.data.length} (Page ${this.currentPage + 1}/${totalPages})`;
    }
    
    navigatePage(direction) {
        const totalPages = Math.ceil(this.data.length / this.promptsPerPage);
        const newPage = this.currentPage + direction;
        if (newPage >= 0 && newPage < totalPages) {
            this.currentPage = newPage;
            this.updateNavigationButtons();
            this.displayPage();
        }
    }
    
    displayPage() {
        if (this.data.length === 0) {
            this.promptText.innerHTML = '<p>No data loaded</p>';
            this.promptInfo.style.display = 'none';
            return;
        }
        
        const startIdx = this.currentPage * this.promptsPerPage;
        const endIdx = Math.min(startIdx + this.promptsPerPage, this.data.length);
        const pagePrompts = this.data.slice(startIdx, endIdx);
        
        let html = '';
        for (let i = 0; i < pagePrompts.length; i++) {
            const prompt = pagePrompts[i];
            html += this.renderPrompt(prompt, i === 0);
        }
        
        this.promptText.innerHTML = html;
        this.promptInfo.style.display = 'none';
    }
    
    renderPrompt(prompt, isFirst = false) {
        const feature = this.featureSelect.value;
        // Handle both data formats: dictionary format (feature 45426) and single value format (other features)
        let maxActivation = 0;
        if (prompt.max_feature_activations && prompt.max_feature_activations[feature]) {
            maxActivation = prompt.max_feature_activations[feature];
        } else if (prompt.max_feature_activation) {
            maxActivation = prompt.max_feature_activation;
        }
        
        // Build prompt info
        let infoHTML = `<strong>Prompt ${prompt.prompt_id}</strong> - Max Activation: ${maxActivation.toFixed(4)}`;
        
        if (prompt.token_type) {
            infoHTML += ` - Token Type: ${prompt.token_type}`;
        }
        
        if (prompt.token_position !== undefined) {
            infoHTML += ` - Position: ${prompt.token_position}`;
        }
        
        if (prompt.max_activation_at_position !== undefined) {
            infoHTML += ` - Position Activation: ${prompt.max_activation_at_position.toFixed(4)}`;
        }
        
        // Build prompt text with highlighting
        let textHTML = '';
        if (!prompt.tokenized_prompt) {
            textHTML = `<p>${this.escapeHtml(prompt.prompt_text)}</p>`;
        } else {
            const tokenizedPrompt = prompt.tokenized_prompt;
            
            // Create activation map from tokens
            const activationMap = new Map();
            if (prompt.tokens) {
                prompt.tokens.forEach(token => {
                    let activation = 0;
                    // Handle both data formats: dictionary format (feature 45426) and single value format (other features)
                    if (token.feature_activations && token.feature_activations[feature]) {
                        activation = token.feature_activations[feature];
                    } else if (token.feature_activation) {
                        activation = token.feature_activation;
                    }
                    
                    if (activation > 0) {
                        activationMap.set(token.position, activation);
                    }
                });
            }
            
            // Generate HTML with highlighting, skip first 2 tokens (<bos> tokens)
            for (let i = 0; i < tokenizedPrompt.length; i++) {
                const token = tokenizedPrompt[i];
                const activation = activationMap.get(i) || 0;
                
                // Display newlines as literal "\n"
                const displayToken = token === '\n' ? '\\n' : token;
                
                if (activation > this.currentThreshold) {
                    const colorClass = this.getActivationColorClass(activation);
                    textHTML += `<span class="token highlighted ${colorClass}">${this.escapeHtml(displayToken)}</span>`;
                } else {
                    textHTML += `<span class="token">${this.escapeHtml(displayToken)}</span>`;
                }
            }
        }
        
        const marginTop = isFirst ? '0' : '20px';
        
        return `
            <div style="margin-top: ${marginTop}; padding: 10px; border-bottom: 1px solid #ddd;">
                <div style="margin-bottom: 10px; font-size: 11px; color: #333;">${infoHTML}</div>
                <div style="line-height: 1.5; font-size: 13px; word-wrap: break-word;">${textHTML}</div>
            </div>
        `;
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
    
    async initializeViewer() {
        this.setStatus('Initializing viewer...', 'loading');
        try {
            await this.discoverFeatures();
            await this.loadFeatureMetadata();
            this.populateFeatureDropdown();
            this.setStatus('Ready - select a feature and click "Load Data"', 'success');
        } catch (error) {
            console.error('Error initializing viewer:', error);
            this.setStatus(`Error initializing viewer: ${error.message}`, 'error');
        }
    }
    
    async discoverFeatures() {
        // Since we can't directly list directories from browser, we'll try to fetch known features
        // This is a workaround - we'll try to fetch the active.jsonl file for each potential feature
        const knownFeatures = [
            '10392', '11383', '21953', '26196', '45426', '57516', '65116', '71187', 
            '74079', '74855', '80134', '8524', '85422', '90900', '91547', '102414', 
            '111921', '128628'
        ];
        
        this.features = [];
        
        for (const feature of knownFeatures) {
            try {
                const response = await fetch(`../gemma_trainer131k-l0-114_layer20/1000_prompts/${feature}/active.jsonl`);
                if (response.ok) {
                    this.features.push(feature);
                }
            } catch (error) {
                // Feature doesn't exist, skip it
                continue;
            }
        }
        
        // Sort features numerically
        this.features.sort((a, b) => parseInt(a) - parseInt(b));
    }
    
    async loadFeatureMetadata() {
        try {
            const response = await fetch('../../../explanations/gemma_trainer131k-l0-114_layer20.csv');
            if (!response.ok) {
                throw new Error(`Failed to load feature metadata: ${response.status}`);
            }
            
            const csvText = await response.text();
            const lines = csvText.trim().split('\n');
            
            // Skip header line
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line) {
                    // Parse CSV line - handle quotes in description
                    // Expected format: feature_id,link,"description"
                    const match = line.match(/^(\d+),([^,]+),"(.*)"$/);
                    if (match) {
                        const [, featureId, link, description] = match;
                        this.featureMetadata.set(featureId, {
                            description: description,
                            link: link
                        });
                    }
                }
            }
        } catch (error) {
            console.warn('Could not load feature metadata:', error);
            // Continue without metadata
        }
    }
    
    populateFeatureDropdown() {
        // Clear existing options
        this.featureSelect.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select a feature...';
        this.featureSelect.appendChild(defaultOption);
        
        // Add feature options
        this.features.forEach(feature => {
            const option = document.createElement('option');
            option.value = feature;
            option.textContent = feature;
            this.featureSelect.appendChild(option);
        });
    }
    
    updateFeatureMetadata() {
        const selectedFeature = this.featureSelect.value;
        
        if (!selectedFeature) {
            this.featureDescription.textContent = 'No feature selected';
            this.neuronpediaLink.style.display = 'none';
            return;
        }
        
        const metadata = this.featureMetadata.get(selectedFeature);
        
        if (metadata) {
            this.featureDescription.textContent = metadata.description;
            this.neuronpediaLink.href = metadata.link;
            this.neuronpediaLink.style.display = 'inline';
        } else {
            this.featureDescription.textContent = `Feature ${selectedFeature} - No description available`;
            // Generate Neuronpedia link even if we don't have metadata
            this.neuronpediaLink.href = `https://www.neuronpedia.org/gemma-2-9b/20-gemmascope-res-131k/${selectedFeature}`;
            this.neuronpediaLink.style.display = 'inline';
        }
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FeatureViewer();
});