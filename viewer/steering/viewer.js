class PCASteeringViewer {
    constructor() {
        this.data = null;
        this.availablePCs = [];
        this.currentDataSource = '';
        this.currentPC = '';
        
        // DOM elements
        this.dataSourceSelect = document.getElementById('data-source');
        this.featureSelect = document.getElementById('feature-select');
        this.loadBtn = document.getElementById('load-btn');
        this.status = document.getElementById('status');
        this.promptsContainer = document.getElementById('prompts-container');
        this.metadata = document.getElementById('metadata');
        this.loadingBar = document.getElementById('loading-bar');
        
        
        // URL parameters
        this.urlParams = new URLSearchParams(window.location.search);
        
        this.initEventListeners();
        this.initializeViewer();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.dataSourceSelect.addEventListener('change', () => this.onDataSourceChange());
        this.featureSelect.addEventListener('change', () => this.updateMetadataDisplay());
    }
    
    async initializeViewer() {
        this.setStatus('Initializing viewer...', 'loading');
        try {
            await this.loadDataSources();
            
            // Check for URL parameters and auto-load if present
            const { dataSource, pc } = this.parseUrlParameters();
            if (dataSource && pc) {
                this.dataSourceSelect.value = dataSource;
                await this.onDataSourceChange();
                this.featureSelect.value = pc;
                this.updateMetadataDisplay();
                await this.loadData();
            } else {
                this.setStatus('Select a data source and PC component to begin', '');
            }
        } catch (error) {
            console.error('Error initializing viewer:', error);
            this.setStatus('Error initializing viewer', 'error');
        }
    }
    
    parseUrlParameters() {
        const dataSourceParam = this.urlParams.get('source');
        const pcParam = this.urlParams.get('pc');
        
        // Map URL parameters to internal data source names
        const dataSourceMapping = {
            'role_shared': 'roles_240'
        };
        
        const dataSource = dataSourceMapping[dataSourceParam] || dataSourceParam;
        const pc = pcParam;
        
        return { dataSource, pc };
    }
    
    updateUrlParameters(dataSource = null, pc = null) {
        const url = new URL(window.location);
        
        if (dataSource) {
            // Map internal names back to URL parameters
            const urlParamMapping = {
                'roles_240': 'role_shared'
            };
            const urlParam = urlParamMapping[dataSource] || dataSource;
            url.searchParams.set('source', urlParam);
        } else {
            url.searchParams.delete('source');
        }
        
        if (pc) {
            url.searchParams.set('pc', pc);
        } else {
            url.searchParams.delete('pc');
        }
        
        window.history.replaceState({}, '', url);
    }
    
    async loadDataSources() {
        const dataSources = [
            { value: 'roles_240', label: 'Roles (Shared Questions)' }
        ];
        
        // Clear and populate data source select
        this.dataSourceSelect.innerHTML = '<option value="">Select data source...</option>';
        
        dataSources.forEach(source => {
            const option = document.createElement('option');
            option.value = source.value;
            option.textContent = source.label;
            this.dataSourceSelect.appendChild(option);
        });
    }
    
    async onDataSourceChange() {
        const dataSource = this.dataSourceSelect.value;
        if (!dataSource) {
            this.featureSelect.innerHTML = '<option value="">Select data source first</option>';
            return;
        }
        
        this.currentDataSource = dataSource;
        this.updateUrlParameters(dataSource, null);
        this.setStatus('Loading PC list...', 'loading');
        this.showLoadingBar();
        
        try {
            const pcs = await this.discoverAvailablePCs(dataSource);
            this.populatePCSelect(pcs);
            this.setStatus(`Found ${pcs.length} PC components`, '');
        } catch (error) {
            console.error('Error loading PC list:', error);
            this.setStatus('Error loading PC list', 'error');
            this.featureSelect.innerHTML = '<option value="">Error loading list</option>';
        } finally {
            this.hideLoadingBar();
        }
    }
    
    async discoverAvailablePCs(dataSource) {
        const pcs = [];
        
        // Check for available PC files in the data source directory
        const knownPCs = ['pc1']; // Add more as they become available
        
        for (const pc of knownPCs) {
            try {
                const response = await fetch(`data/${dataSource}/${pc}.json`, { method: 'HEAD' });
                if (response.ok) {
                    pcs.push({
                        value: pc,
                        label: `${pc.toUpperCase()} (Principal Component)`,
                        filename: `${pc}.json`
                    });
                }
            } catch (e) {
                console.warn(`PC file ${pc}.json not found in ${dataSource}`);
            }
        }
        
        if (pcs.length === 0) {
            // Fallback if no files found
            pcs.push({
                value: 'pc1',
                label: 'PC1 (Principal Component)',
                filename: 'pc1.json'
            });
        }
        
        return pcs;
    }
    
    populatePCSelect(pcs) {
        this.availablePCs = pcs;
        
        // Clear existing options
        this.featureSelect.innerHTML = '';
        
        if (pcs.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No PC components available';
            this.featureSelect.appendChild(option);
            return;
        }
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select a PC component...';
        this.featureSelect.appendChild(defaultOption);
        
        // Add PC options
        pcs.forEach(pc => {
            const option = document.createElement('option');
            option.value = pc.value;
            option.textContent = pc.label;
            this.featureSelect.appendChild(option);
        });
    }
    
    updateMetadataDisplay() {
        const selectedValue = this.featureSelect.value;
        if (!selectedValue) {
            this.metadata.style.display = 'none';
            return;
        }
        
        const selectedPC = this.availablePCs.find(p => p.value === selectedValue);
        if (!selectedPC) {
            this.metadata.style.display = 'none';
            return;
        }
        
        this.metadata.style.display = 'block';
        
        // Update metadata info for PC component
        const featureInfo = document.getElementById('feature-info');
        const featureDescriptions = document.getElementById('feature-descriptions');
        const neuronpediaLinks = document.getElementById('neuronpedia-links');
        
        if (featureInfo) {
            featureInfo.innerHTML = `<strong>${selectedPC.label}</strong>`;
        }
        
        if (featureDescriptions) {
            let description = '';
            if (selectedPC.value === 'pc1') {
                description = '<strong>Description:</strong><br>Assistant-like to role-playing.';
            } else {
                description = `<strong>Description:</strong><br>Principal Component ${selectedPC.value.slice(-1)} from role steering analysis.`;
            }
            featureDescriptions.innerHTML = description;
        }
        
        if (neuronpediaLinks) {
            neuronpediaLinks.innerHTML = '';
        }
        
        // Update URL parameter
        this.updateUrlParameters(this.currentDataSource, selectedPC.value);
    }
    
    async loadData() {
        const dataSource = this.dataSourceSelect.value;
        const selectedValue = this.featureSelect.value;
        
        if (!dataSource || !selectedValue) {
            this.setStatus('Please select both data source and PC component', 'error');
            return;
        }
        
        const selectedPC = this.availablePCs.find(p => p.value === selectedValue);
        if (!selectedPC) {
            this.setStatus('Invalid PC component selected', 'error');
            return;
        }
        
        this.currentPC = selectedValue;
        this.updateUrlParameters(dataSource, selectedValue);
        this.setStatus('Loading data...', 'loading');
        this.showLoadingBar();
        this.loadBtn.disabled = true;
        
        try {
            const response = await fetch(`data/${dataSource}/${selectedPC.filename}`);
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
        if (!this.data) {
            this.clearDisplay();
            return;
        }
        
        let html = '';
        
        // Get all prompts for this PC
        const prompts = Object.keys(this.data);
        
        prompts.forEach(prompt => {
            const magnitudeData = this.data[prompt];
            html += this.renderPromptRow(prompt, magnitudeData);
        });
        
        this.promptsContainer.innerHTML = html;
        
        // Initialize response navigation after rendering
        setTimeout(() => this.initResponseNavigation(), 0);
    }
    
    renderPromptRow(prompt, magnitudeData) {
        let scrollableHTML = '';
        
        // Sort magnitudes with 0 first, then numerically
        const magnitudes = Object.keys(magnitudeData)
            .sort((a, b) => {
                const aNum = parseFloat(a);
                const bNum = parseFloat(b);
                if (aNum === 0) return -1;
                if (bNum === 0) return 1;
                return aNum - bNum;
            });
        
        // Add steering responses for each magnitude
        magnitudes.forEach(magnitude => {
            const responseList = magnitudeData[magnitude];
            const magnitudeValue = parseFloat(magnitude);
            const label = magnitudeValue === 0 ? 'Default Response' : `Magnitude ${magnitude}`;
            scrollableHTML += this.renderResponseBox(label, responseList, magnitudeValue);
        });
        
        return `
            <div class="prompt-row">
                <div class="prompt-question"><b>Prompt:</b> ${this.escapeHtml(prompt)}</div>
                <div class="responses-layout">
                    <div class="scrollable-responses-container">
                        ${scrollableHTML}
                    </div>
                </div>
            </div>
        `;
    }
    
    renderResponseBox(label, responseList, magnitudeValue) {
        if (!responseList || responseList.length === 0) return '';
        
        const currentResponse = responseList[0]; // Start with first response
        const processedResponse = this.processResponseText(currentResponse);
        
        // Determine color class based on magnitude
        let colorClass;
        if (magnitudeValue > 0) {
            colorClass = 'steering-positive';
        } else if (magnitudeValue < 0) {
            colorClass = 'steering-negative';
        } else {
            colorClass = 'steering-zero';
        }
        
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
            <div class="response-box ${colorClass}" id="${boxId}" data-responses='${JSON.stringify(responseList).replace(/'/g, "&#39;")}' data-current-index="0">
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
        
        // Replace newlines with <br> tags
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
            const viewer = window.pcaSteeringViewer;
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
    window.pcaSteeringViewer = new PCASteeringViewer();
});