class SteeringViewer {
    constructor() {
        this.data = null;
        this.availableFeatures = [];
        this.featureMetadata = new Map();
        
        // DOM elements
        this.featureSelect = document.getElementById('feature-select');
        this.loadBtn = document.getElementById('load-btn');
        this.status = document.getElementById('status');
        this.promptsContainer = document.getElementById('prompts-container');
        this.metadata = document.getElementById('metadata');
        this.featureInfo = document.getElementById('feature-info');
        this.featureDescriptions = document.getElementById('feature-descriptions');
        this.neuronpediaLinks = document.getElementById('neuronpedia-links');
        this.loadingBar = document.getElementById('loading-bar');
        
        // Visibility controls
        this.toggleDefault = document.getElementById('toggle-default');
        this.toggleAblation = document.getElementById('toggle-ablation');
        this.toggleSteering = document.getElementById('toggle-steering');
        
        // Visibility state
        this.showDefault = true;
        this.showAblation = true;
        this.showSteering = true;
        
        this.initEventListeners();
        this.initializeViewer();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.featureSelect.addEventListener('change', () => this.updateMetadataDisplay());
        
        // Visibility controls
        this.toggleDefault.addEventListener('click', () => this.toggleVisibility('default'));
        this.toggleAblation.addEventListener('click', () => this.toggleVisibility('ablation'));
        this.toggleSteering.addEventListener('click', () => this.toggleVisibility('steering'));
    }
    
    async initializeViewer() {
        this.setStatus('Initializing viewer...', 'loading');
        try {
            await this.discoverAvailableFiles();
            await this.loadFeatureMetadata();
            this.populateFeatureSelect();
            
            // Check for URL parameters
            const urlParams = new URLSearchParams(window.location.search);
            const featureId = urlParams.get('feature_id');
            const featureGroup = urlParams.get('feature_group');
            
            if (featureId || featureGroup) {
                const paramValue = featureId || featureGroup;
                if (this.availableFeatures.some(f => f.value === paramValue)) {
                    this.featureSelect.value = paramValue;
                    this.updateMetadataDisplay();
                    // Auto-load if URL parameter is present
                    setTimeout(() => this.loadData(), 100);
                }
            }
            
            this.setStatus('Select a feature or group to begin', '');
        } catch (error) {
            console.error('Error initializing viewer:', error);
            this.setStatus('Error initializing viewer', 'error');
        }
    }
    
    async discoverAvailableFiles() {
        const basePath = '../gemma_trainer131k-l0-114_layer20/';
        this.availableFeatures = [];
        
        // Known features and groups (updated by update_features.py script)
        const knownFeatures = [
                '45426', '116246'
            ];
        
        const knownGroupFiles = [
                'ai_introspective_diff.json', 'top_mean_diff.json', 'transitions.json'
            ];
        
        // Try to discover individual feature files and group files
        try {
            // Check for known individual files first
            for (const feature of knownFeatures) {
                try {
                    const response = await fetch(`${basePath}${feature}.json`, { method: 'HEAD' });
                    if (response.ok) {
                        this.availableFeatures.push({
                            value: feature,
                            label: `Feature ${feature}`,
                            type: 'individual',
                            filename: `${feature}.json`
                        });
                    }
                } catch (e) {
                    // File doesn't exist, skip
                }
            }
            
            // Check for group files
            for (const filename of knownGroupFiles) {
                try {
                    const response = await fetch(`${basePath}${filename}`, { method: 'HEAD' });
                    if (response.ok) {
                        // Try to get just the metadata to determine the readable name
                        const dataResponse = await fetch(`${basePath}${filename}`);
                        const data = await dataResponse.json();
                        
                        this.availableFeatures.push({
                            value: filename.replace('.json', ''),
                            label: data.readable_group_name || filename.replace('.json', ''),
                            type: 'group',
                            filename: filename,
                            groupData: data
                        });
                    }
                } catch (e) {
                    // File doesn't exist, skip
                }
            }
            
        } catch (error) {
            console.warn('Error discovering files:', error);
            // Fallback to hardcoded features
            this.availableFeatures = [
                {
                    value: '45426',
                    label: 'Feature 45426',
                    type: 'individual',
                    filename: '45426.json'
                }
            ];
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
            
            // Skip header line (feature_id,link,claude_desc)
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line) {
                    // Parse CSV properly handling quoted descriptions
                    const parts = this.parseCSVLine(line);
                    if (parts.length >= 3) {
                        const featureId = parts[0].trim();
                        const link = parts[1].trim();
                        const description = parts[2].trim().replace(/^"|"$/g, ''); // Remove quotes
                        
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
    
    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
                    // Escaped quote - add literal quote and skip next character
                    current += '"';
                    i++; // Skip the next quote
                } else {
                    // Toggle quote state
                    inQuotes = !inQuotes;
                }
            } else if (char === ',' && !inQuotes) {
                result.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        
        result.push(current);
        return result;
    }
    
    populateFeatureSelect() {
        // Clear existing options
        this.featureSelect.innerHTML = '';
        
        if (this.availableFeatures.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No features available';
            this.featureSelect.appendChild(option);
            return;
        }
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select a feature or group...';
        this.featureSelect.appendChild(defaultOption);
        
        // Add features
        this.availableFeatures.forEach(feature => {
            const option = document.createElement('option');
            option.value = feature.value;
            
            // Include description in dropdown if available for individual features
            if (feature.type === 'individual') {
                const metadata = this.featureMetadata.get(feature.value);
                if (metadata && metadata.description) {
                    const maxLength = 60; // Maximum characters for description
                    let description = metadata.description;
                    if (description.length > maxLength) {
                        description = description.substring(0, maxLength) + '...';
                    }
                    option.textContent = `${feature.value} - ${description}`;
                } else {
                    option.textContent = feature.label;
                }
            } else {
                // For group features, just use the label
                option.textContent = feature.label;
            }
            
            this.featureSelect.appendChild(option);
        });
    }
    
    updateMetadataDisplay() {
        const selectedValue = this.featureSelect.value;
        if (!selectedValue) {
            this.metadata.style.display = 'none';
            return;
        }
        
        const selectedFeature = this.availableFeatures.find(f => f.value === selectedValue);
        if (!selectedFeature) {
            this.metadata.style.display = 'none';
            return;
        }
        
        this.metadata.style.display = 'block';
        
        if (selectedFeature.type === 'individual') {
            this.displayIndividualFeatureMetadata(selectedFeature);
        } else if (selectedFeature.type === 'group') {
            this.displayGroupMetadata(selectedFeature);
        }
        
        // Update URL parameter
        this.updateUrlParameter(selectedFeature);
    }
    
    displayIndividualFeatureMetadata(feature) {
        const featureId = feature.value;
        const metadata = this.featureMetadata.get(featureId);
        
        this.featureInfo.innerHTML = `<strong>Feature ${featureId}</strong>`;
        
        if (metadata && metadata.description) {
            this.featureDescriptions.innerHTML = `<strong>Description:</strong><br>${metadata.description}`;
        } else {
            this.featureDescriptions.innerHTML = `<strong>Description:</strong><br>No description available`;
        }
        
        const neuronpediaUrl = metadata ? metadata.link : `https://www.neuronpedia.org/gemma-2-9b/20-gemmascope-res-131k/${featureId}`;
        this.neuronpediaLinks.innerHTML = `<a href="${neuronpediaUrl}" target="_blank">View on Neuronpedia</a>`;
    }
    
    displayGroupMetadata(feature) {
        const groupData = feature.groupData;
        
        this.featureInfo.innerHTML = `<strong>${groupData.readable_group_name}</strong>`;
        
        // Display all features in the group with descriptions
        let descriptionsHtml = '<div class="feature-descriptions">';
        
        groupData.feature_id.forEach(featureId => {
            const metadata = this.featureMetadata.get(featureId.toString());
            const description = metadata ? metadata.description : 'No description available';
            const neuronpediaUrl = metadata ? metadata.link : `https://www.neuronpedia.org/gemma-2-9b/20-gemmascope-res-131k/${featureId}`;
            
            descriptionsHtml += `
                <div class="feature-description-item">
                    <strong>Feature ${featureId}:</strong> ${description}<br>
                    <a href="${neuronpediaUrl}" target="_blank">View on Neuronpedia</a>
                </div>
            `;
        });
        
        descriptionsHtml += '</div>';
        this.featureDescriptions.innerHTML = descriptionsHtml;
        
        // Clear the neuronpedia links section since they're now inline
        this.neuronpediaLinks.innerHTML = '';
    }
    
    updateUrlParameter(feature) {
        const url = new URL(window.location);
        
        // Clear existing parameters
        url.searchParams.delete('feature_id');
        url.searchParams.delete('feature_group');
        
        // Add appropriate parameter
        if (feature.type === 'individual') {
            url.searchParams.set('feature_id', feature.value);
        } else if (feature.type === 'group') {
            url.searchParams.set('feature_group', feature.value);
        }
        
        window.history.replaceState({}, '', url);
    }
    
    async loadData() {
        const selectedValue = this.featureSelect.value;
        if (!selectedValue) {
            this.setStatus('Please select a feature or group', 'error');
            return;
        }
        
        const selectedFeature = this.availableFeatures.find(f => f.value === selectedValue);
        if (!selectedFeature) {
            this.setStatus('Invalid feature selected', 'error');
            return;
        }
        
        this.setStatus('Loading data...', 'loading');
        this.showLoadingBar();
        this.loadBtn.disabled = true;
        
        try {
            const response = await fetch(`../gemma_trainer131k-l0-114_layer20/${selectedFeature.filename}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.data = await response.json();
            this.setStatus('Data loaded successfully', 'success');
            
            this.displayData();
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.setStatus(`Error loading data: ${error.message}`, 'error');
            this.data = null;
            this.clearDisplay();
        } finally {
            this.hideLoadingBar();
            this.loadBtn.disabled = false;
        }
    }
    
    showLoadingBar() {
        this.loadingBar.classList.add('active');
    }
    
    hideLoadingBar() {
        this.loadingBar.classList.remove('active');
    }
    
    setStatus(message, type = '') {
        this.status.textContent = message;
        this.status.className = `status ${type}`;
    }
    
    displayData() {
        if (!this.data || !this.data.results) {
            this.clearDisplay();
            return;
        }
        
        let html = '';
        
        // Get all prompts for this feature/group
        const prompts = Object.keys(this.data.results);
        
        prompts.forEach(prompt => {
            const responses = this.data.results[prompt];
            html += this.renderPromptRow(prompt, responses);
        });
        
        this.promptsContainer.innerHTML = html;
        
        // Initialize response navigation after rendering
        setTimeout(() => this.initResponseNavigation(), 0);
    }
    
    renderPromptRow(prompt, responses) {
        let defaultHTML = '';
        let scrollableHTML = '';
        
        // 1. Add default (0.0) first - this will be frozen
        if (this.showDefault && responses.steering && responses.steering['0.0']) {
            defaultHTML = this.renderResponseBox('Default Response', responses.steering['0.0'], 'default');
        }
        
        // 2. Add ablation responses - these will scroll
        if (this.showAblation && responses.ablation) {

            if (responses.ablation.mean_ablation) {
                scrollableHTML += this.renderResponseBox('Mean Ablation', responses.ablation.mean_ablation, 'ablation');
            }

            if (responses.ablation.mean_ablation_20_end) {
                scrollableHTML += this.renderResponseBox('Mean Ablation Target Layer to End', responses.ablation.mean_ablation_20_end, 'ablation');
            }
            
            if (responses.ablation.add_error) {
                scrollableHTML += this.renderResponseBox('Zero Ablation via SAE', responses.ablation.add_error, 'ablation');
            }
            
            if (responses.ablation.projection_zero_ablate) {
                scrollableHTML += this.renderResponseBox('Zero Ablation via Projection', responses.ablation.projection_zero_ablate, 'ablation');
            }
        }
        
        // 3. Add steering responses in increasing order (excluding 0.0 which we already added) - these will scroll
        if (this.showSteering && responses.steering) {
            const magnitudes = Object.keys(responses.steering)
                .filter(mag => mag !== '0.0')
                .sort((a, b) => parseFloat(a) - parseFloat(b));
            
            magnitudes.forEach(magnitude => {
                const responseList = responses.steering[magnitude];
                const label = `Steering ${magnitude}`;
                scrollableHTML += this.renderResponseBox(label, responseList, 'steering');
            });
        }
        
        return `
            <div class="prompt-row">
                <div class="prompt-question"><b>Prompt:</b> ${this.escapeHtml(prompt)}</div>
                <div class="responses-layout">
                    <div class="default-response-container">
                        ${defaultHTML}
                    </div>
                    <div class="scrollable-responses-container">
                        ${scrollableHTML}
                    </div>
                </div>
            </div>
        `;
    }
    
    renderResponseBox(label, responseList, type) {
        if (!responseList || responseList.length === 0) return '';
        
        const currentResponse = responseList[0]; // Start with first response
        const processedResponse = this.processResponseText(currentResponse);
        
        // Create navigation - always show counter, only show buttons if multiple responses
        let navigationHTML = '';
        if (responseList.length > 1) {
            navigationHTML = `
                <div class="response-navigation">
                    <button class="nav-btn prev-btn" onclick="this.closest('.response-box').changeResponse(-1)" disabled>‹</button>
                    <span class="response-counter">1 / ${responseList.length}</span>
                    <button class="nav-btn next-btn" onclick="this.closest('.response-box').changeResponse(1)">›</button>
                </div>
            `;
        } else {
            navigationHTML = `
                <div class="response-navigation">
                    <span class="response-counter">1 / 1</span>
                </div>
            `;
        }
        
        const boxId = `response-${Math.random().toString(36).substr(2, 9)}`;
        
        return `
            <div class="response-box ${type}" id="${boxId}" data-responses='${JSON.stringify(responseList).replace(/'/g, "&#39;")}' data-current-index="0">
                <div class="response-magnitude">
                    ${label}
                </div>
                <div class="response-text">
                    ${processedResponse}
                </div>
                ${navigationHTML}
            </div>
        `;
    }
    
    processResponseText(text) {
        // Trim leading and trailing whitespace
        let processed = text.trim();
        
        // Escape HTML
        processed = this.escapeHtml(processed);
        
        // Handle basic markdown formatting
        processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        processed = processed.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Replace newlines with <br> tags to avoid pre-line whitespace issues
        processed = processed.replace(/\n/g, '<br>');
        
        return processed;
    }
    
    clearDisplay() {
        this.promptsContainer.innerHTML = `
            <div style="text-align: center; color: #666; margin-top: 50px;">
                No data loaded
            </div>
        `;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    toggleVisibility(type) {
        switch(type) {
            case 'default':
                this.showDefault = !this.showDefault;
                this.toggleDefault.textContent = this.showDefault ? 'Hide Default' : 'Show Default';
                this.toggleDefault.classList.toggle('active', this.showDefault);
                break;
            case 'ablation':
                this.showAblation = !this.showAblation;
                this.toggleAblation.textContent = this.showAblation ? 'Hide Ablation' : 'Show Ablation';
                this.toggleAblation.classList.toggle('active', this.showAblation);
                break;
            case 'steering':
                this.showSteering = !this.showSteering;
                this.toggleSteering.textContent = this.showSteering ? 'Hide Steering' : 'Show Steering';
                this.toggleSteering.classList.toggle('active', this.showSteering);
                break;
        }
        
        // Re-render the data with new visibility settings
        this.displayData();
    }
    
    // Add method to handle response navigation
    initResponseNavigation() {
        // Add change response method to response boxes
        const changeResponse = function(direction) {
            const responses = JSON.parse(this.getAttribute('data-responses'));
            let currentIndex = parseInt(this.getAttribute('data-current-index'));
            
            currentIndex += direction;
            if (currentIndex < 0) currentIndex = 0;
            if (currentIndex >= responses.length) currentIndex = responses.length - 1;
            
            this.setAttribute('data-current-index', currentIndex);
            
            // Update response text
            const responseText = this.querySelector('.response-text');
            const viewer = window.steeringViewer;
            responseText.innerHTML = viewer.processResponseText(responses[currentIndex]);
            
            // Update navigation buttons
            const prevBtn = this.querySelector('.prev-btn');
            const nextBtn = this.querySelector('.next-btn');
            const counter = this.querySelector('.response-counter');
            
            if (prevBtn) prevBtn.disabled = currentIndex === 0;
            if (nextBtn) nextBtn.disabled = currentIndex === responses.length - 1;
            if (counter) counter.textContent = `${currentIndex + 1} / ${responses.length}`;
        };
        
        // Attach method to all response boxes
        document.querySelectorAll('.response-box').forEach(box => {
            box.changeResponse = changeResponse;
        });
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.steeringViewer = new SteeringViewer();
});