class SteeringViewer {
    constructor() {
        this.data = null;
        this.availableFeatures = [];
        
        // DOM elements
        this.featureSelect = document.getElementById('feature-select');
        this.loadBtn = document.getElementById('load-btn');
        this.status = document.getElementById('status');
        this.promptsContainer = document.getElementById('prompts-container');
        
        // Visibility controls
        this.toggleDefault = document.getElementById('toggle-default');
        this.toggleAblation = document.getElementById('toggle-ablation');
        this.toggleSteering = document.getElementById('toggle-steering');
        
        // Visibility state
        this.showDefault = true;
        this.showAblation = true;
        this.showSteering = true;
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.featureSelect.addEventListener('change', () => this.displayData());
        
        // Visibility controls
        this.toggleDefault.addEventListener('click', () => this.toggleVisibility('default'));
        this.toggleAblation.addEventListener('click', () => this.toggleVisibility('ablation'));
        this.toggleSteering.addEventListener('click', () => this.toggleVisibility('steering'));
    }
    
    async loadData() {
        this.setStatus('Loading data...', 'loading');
        this.loadBtn.disabled = true;
        
        try {
            const selectedFeature = this.featureSelect.value;
            const response = await fetch(`../gemma_trainer131k-l0-114_layer20/${selectedFeature}.json`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.data = await response.json();
            this.setStatus(`Loaded steering results for feature ${selectedFeature}`, 'success');
            this.displayData();
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.setStatus(`Error loading data: ${error.message}`, 'error');
            this.data = null;
            this.clearDisplay();
        } finally {
            this.loadBtn.disabled = false;
        }
    }
    
    populateFeatureSelect() {
        // Feature select is now hardcoded in HTML since we load individual files
        // Keep this method for compatibility
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
        
        // Get all prompts for this feature
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