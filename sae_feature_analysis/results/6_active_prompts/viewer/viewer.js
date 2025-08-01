class FeatureViewer {
    constructor() {
        this.data = [];
        this.currentPage = 0;
        this.promptsPerPage = 50;
        this.currentThreshold = 0;
        this.features = [];
        this.featureMetadata = new Map();
        this.featuresIndex = null; // Store the features index
        this.promptsData = null; // Store all prompts data
        this.loadedFeatures = new Map(); // Cache for loaded feature data
        
        // DOM elements
        this.featureSelect = document.getElementById('feature-select');
        this.tokenTypeSelect = document.getElementById('token-type-select');
        this.activeTypeSelect = document.getElementById('active-type-select');
        this.thresholdSlider = document.getElementById('threshold-slider');
        this.thresholdValue = document.getElementById('threshold-value');
        this.loadBtn = document.getElementById('load-btn');
        this.sortOrderSelect = document.getElementById('sort-order-select');
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
        this.loadingProgress = document.getElementById('loading-progress');
        
        this.initEventListeners();
        this.initializeViewer();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.thresholdSlider.addEventListener('input', (e) => this.updateThreshold(e.target.value));
        this.prevBtn.addEventListener('click', () => this.navigatePage(-1));
        this.nextBtn.addEventListener('click', () => this.navigatePage(1));
        this.featureSelect.addEventListener('change', () => {
            this.updateFeatureMetadata();
            this.updateUrlParameter();
        });
        this.sortOrderSelect.addEventListener('change', () => this.applySortOrder());
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
        
        if (!this.featuresIndex) {
            this.setStatus('Features index not loaded', 'error');
            return;
        }
        
        this.setStatus('Filtering data...', 'loading');
        this.loadBtn.disabled = true;
        
        try {
            // Load individual feature data if not cached
            let featureData = this.loadedFeatures.get(feature);
            if (!featureData) {
                this.setStatus(`Loading feature ${feature} data...`, 'loading');
                const response = await fetch(`../gemma_trainer131k-l0-114_layer20/1000_prompts/features/${feature}.json`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                featureData = await response.json();
                this.loadedFeatures.set(feature, featureData);
            }
            
            // Get the appropriate data subset
            let rawData = featureData[activeType][tokenType] || [];
            
            // Store the raw activation data for lazy loading
            this.currentActivationData = rawData;
            this.data = []; // Will be populated by displayPage()
            
            this.currentPage = 0;
            
            // Load prompts data if not already loaded
            await this.loadPromptsData();
            
            this.applySortOrder();
            this.setStatus(`Prepared ${rawData.length} prompts for lazy loading`, 'success');
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
    
    
    applySortOrder() {
        if (!this.currentActivationData || this.currentActivationData.length === 0) {
            return;
        }
        
        const sortOrder = this.sortOrderSelect.value;
        
        if (sortOrder === 'activation') {
            // Sort by max activation in descending order
            this.currentActivationData.sort((a, b) => {
                const aActivation = Math.max(...(a.tokens || []).map(t => t.act || 0), 0);
                const bActivation = Math.max(...(b.tokens || []).map(t => t.act || 0), 0);
                return bActivation - aActivation;
            });
        } else {
            // For 'default' order, sort by prompt_id to restore original order
            this.currentActivationData.sort((a, b) => a.id - b.id);
        }
        
        // Reset to first page and update display
        this.currentPage = 0;
        this.updateNavigation();
        this.displayPage();
    }
    
    setStatus(message, type = '') {
        this.status.textContent = message;
        this.status.className = `status ${type}`;
    }
    
    updateNavigation() {
        if (this.currentActivationData && this.currentActivationData.length > 0) {
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
        const totalItems = this.currentActivationData ? this.currentActivationData.length : 0;
        const totalPages = Math.ceil(totalItems / this.promptsPerPage);
        this.prevBtn.disabled = this.currentPage === 0;
        this.nextBtn.disabled = this.currentPage === totalPages - 1;
        
        const startPrompt = this.currentPage * this.promptsPerPage + 1;
        const endPrompt = Math.min((this.currentPage + 1) * this.promptsPerPage, totalItems);
        this.promptCounter.textContent = `${startPrompt}-${endPrompt} of ${totalItems} (Page ${this.currentPage + 1}/${totalPages})`;
    }
    
    navigatePage(direction) {
        const totalItems = this.currentActivationData ? this.currentActivationData.length : 0;
        const totalPages = Math.ceil(totalItems / this.promptsPerPage);
        const newPage = this.currentPage + direction;
        if (newPage >= 0 && newPage < totalPages) {
            this.currentPage = newPage;
            this.updateNavigationButtons();
            this.displayPage();
        }
    }
    
    displayPage() {
        if (!this.currentActivationData || this.currentActivationData.length === 0) {
            this.promptText.innerHTML = '<p>No data loaded</p>';
            this.promptInfo.style.display = 'none';
            return;
        }
        
        const startIdx = this.currentPage * this.promptsPerPage;
        const endIdx = Math.min(startIdx + this.promptsPerPage, this.currentActivationData.length);
        const pageActivations = this.currentActivationData.slice(startIdx, endIdx);
        
        // Convert activation data to full prompt format for current page
        const pagePrompts = pageActivations.map(entry => {
            const promptData = this.promptsData[entry.id];
            if (!promptData) {
                console.warn(`Prompt ${entry.id} not found in prompts data`);
                return null;
            }
            
            return {
                prompt_id: entry.id,
                tokenized_prompt: promptData.tokens,
                max_feature_activation: 0, // Will be calculated from tokens if needed
                tokens: entry.tokens ? entry.tokens.map(t => ({
                    position: t.pos,
                    feature_activation: t.act
                })) : []
            };
        }).filter(p => p !== null);
        
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
        // Calculate max activation from token data
        let maxActivation = 0;
        if (prompt.tokens && prompt.tokens.length > 0) {
            maxActivation = Math.max(...prompt.tokens.map(token => token.feature_activation || 0));
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
        this.setStatus('Loading data bundle...', 'loading');
        try {
            // Load the consolidated data bundle
            await this.loadFeaturesIndex();
            await this.loadFeatureMetadata();
            this.populateFeatureDropdown();
            
            // Check for feature_id URL parameter
            const urlParams = new URLSearchParams(window.location.search);
            const featureId = urlParams.get('feature_id');
            
            if (featureId) {
                this.handleUrlFeature(featureId);
            } else {
                this.setStatus('Ready - select a feature and click "Load Data"', 'success');
            }
        } catch (error) {
            console.error('Error initializing viewer:', error);
            this.setStatus(`Error initializing viewer: ${error.message}`, 'error');
        }
    }
    
    async loadFeaturesIndex() {
        this.setStatus('Loading features index...', 'loading');
        this.loadingProgress.style.display = 'block';
        this.loadingProgress.textContent = 'Downloading features index...';
        
        try {
            const response = await fetch('features_index.json');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.loadingProgress.textContent = 'Parsing features index...';
            this.featuresIndex = await response.json();
            
            // Extract features list from index
            this.features = this.featuresIndex.features.map(f => f.id).sort((a, b) => parseInt(a) - parseInt(b));
            
            this.loadingProgress.style.display = 'none';
            this.setStatus(`Loaded ${this.featuresIndex.metadata.total_features} features index`, 'success');
        } catch (error) {
            this.loadingProgress.style.display = 'none';
            console.error('Error loading features index:', error);
            throw new Error(`Failed to load features index: ${error.message}`);
        }
    }
    
    async loadPromptsData() {
        if (this.promptsData) {
            return; // Already loaded
        }
        
        this.setStatus('Loading prompts data...', 'loading');
        this.loadingProgress.style.display = 'block';  
        this.loadingProgress.textContent = 'Downloading prompts...';
        
        try {
            const response = await fetch('prompts.json');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.loadingProgress.textContent = 'Parsing prompts JSON...';
            this.promptsData = await response.json();
            
            this.loadingProgress.style.display = 'none';
            this.setStatus(`Loaded ${Object.keys(this.promptsData).length} prompts`, 'success');
        } catch (error) {
            this.loadingProgress.style.display = 'none';
            console.error('Error loading prompts data:', error);
            throw new Error(`Failed to load prompts data: ${error.message}`);
        }
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
                    // Parse CSV line - handle both quoted and unquoted descriptions
                    // Expected formats: feature_id,link,"description" or feature_id,link,description
                    let match = line.match(/^(\d+),([^,]+),"(.*)"$/);
                    if (!match) {
                        // Try without quotes
                        match = line.match(/^(\d+),([^,]+),(.*)$/);
                    }
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
            
            // Include description in dropdown if available
            const metadata = this.featureMetadata.get(feature);
            if (metadata && metadata.description) {
                const maxLength = 60; // Maximum characters for description
                let description = metadata.description;
                if (description.length > maxLength) {
                    description = description.substring(0, maxLength) + '...';
                }
                option.textContent = `${feature} - ${description}`;
            } else {
                option.textContent = feature;
            }
            
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
    
    handleUrlFeature(featureId) {
        if (this.features.includes(featureId)) {
            // Feature is available - select it
            this.featureSelect.value = featureId;
            this.updateFeatureMetadata();
            this.setStatus(`Feature ${featureId} loaded from URL - click "Load Data" to view prompts`, 'success');
        } else {
            // Feature is not available - show error
            this.setStatus(`Feature ${featureId} is not available`, 'error');
        }
    }
    
    updateUrlParameter() {
        const selectedFeature = this.featureSelect.value;
        const url = new URL(window.location);
        
        if (selectedFeature) {
            url.searchParams.set('feature_id', selectedFeature);
        } else {
            url.searchParams.delete('feature_id');
        }
        
        window.history.replaceState({}, '', url);
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FeatureViewer();
});