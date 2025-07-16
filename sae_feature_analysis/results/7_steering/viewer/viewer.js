class SteeringViewer {
    constructor() {
        this.data = null;
        this.availableFeatures = [];
        
        // DOM elements
        this.featureSelect = document.getElementById('feature-select');
        this.loadBtn = document.getElementById('load-btn');
        this.status = document.getElementById('status');
        this.promptsContainer = document.getElementById('prompts-container');
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.featureSelect.addEventListener('change', () => this.displayData());
    }
    
    async loadData() {
        this.setStatus('Loading data...', 'loading');
        this.loadBtn.disabled = true;
        
        try {
            const response = await fetch('../gemma_trainer131k-l0-114_layer20/steering_results.json');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.data = await response.json();
            this.populateFeatureSelect();
            this.setStatus(`Loaded steering results for ${Object.keys(this.data.results).length} features`, 'success');
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
        if (!this.data || !this.data.results) return;
        
        this.availableFeatures = Object.keys(this.data.results);
        this.featureSelect.innerHTML = '';
        
        this.availableFeatures.forEach(feature => {
            const option = document.createElement('option');
            option.value = feature;
            option.textContent = feature;
            this.featureSelect.appendChild(option);
        });
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
        
        const selectedFeature = this.featureSelect.value;
        const featureData = this.data.results[selectedFeature];
        
        if (!featureData) {
            this.clearDisplay();
            return;
        }
        
        let html = '';
        
        // Get all prompts for this feature
        const prompts = Object.keys(featureData);
        
        prompts.forEach(prompt => {
            const responses = featureData[prompt];
            html += this.renderPromptRow(prompt, responses);
        });
        
        this.promptsContainer.innerHTML = html;
    }
    
    renderPromptRow(prompt, responses) {
        // Sort magnitudes numerically
        const magnitudes = Object.keys(responses).sort((a, b) => parseFloat(a) - parseFloat(b));
        
        let responsesHTML = '';
        magnitudes.forEach(magnitude => {
            const response = responses[magnitude];
            responsesHTML += this.renderResponseBox(magnitude, response);
        });
        
        return `
            <div class="prompt-row">
                <div class="prompt-question"><b>Prompt:</b> ${this.escapeHtml(prompt)}</div>
                <div class="responses-container">
                    ${responsesHTML}
                </div>
            </div>
        `;
    }
    
    renderResponseBox(magnitude, response) {
        const magnitudeFloat = parseFloat(magnitude);
        
        // Format magnitude display
        const magnitudeDisplay = magnitudeFloat >= 0 ? `+${magnitude}` : magnitude;
        
        // Process response text to handle newlines and basic markdown
        const processedResponse = this.processResponseText(response);
        
        return `
            <div class="response-box">
                <div class="response-magnitude">
                    Magnitude: ${magnitudeDisplay}
                </div>
                <div class="response-text">
                    ${processedResponse}
                </div>
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
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SteeringViewer();
});