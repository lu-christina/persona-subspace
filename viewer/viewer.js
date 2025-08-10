class TraitsRolesViewer {
    constructor() {
        this.data = {
            responses: [],
            scores: {}
        };
        this.filteredData = [];
        this.currentDataSource = '';
        this.currentTraitRole = '';
        
        // DOM elements
        this.dataSourceSelect = document.getElementById('data-source');
        this.traitRoleSelect = document.getElementById('trait-role-select');
        this.loadBtn = document.getElementById('load-btn');
        this.status = document.getElementById('status');
        this.responsesContainer = document.getElementById('responses-container');
        this.loadingBar = document.getElementById('loading-bar');
        this.stats = document.getElementById('stats');
        
        // Filter elements
        this.filterPos = document.getElementById('filter-pos');
        this.filterNeg = document.getElementById('filter-neg');
        this.filterDefault = document.getElementById('filter-default');
        this.promptIndexSelect = document.getElementById('prompt-index-select');
        this.questionIndexSelect = document.getElementById('question-index-select');
        
        // Sort elements
        this.sortBy = document.getElementById('sort-by');
        this.sortOrder = document.getElementById('sort-order');
        
        this.initEventListeners();
        this.initializeViewer();
    }
    
    initEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.dataSourceSelect.addEventListener('change', () => this.onDataSourceChange());
        
        // Filter change listeners
        this.filterPos.addEventListener('change', () => this.applyFilters());
        this.filterNeg.addEventListener('change', () => this.applyFilters());
        this.filterDefault.addEventListener('change', () => this.applyFilters());
        this.promptIndexSelect.addEventListener('change', () => this.applyFilters());
        this.questionIndexSelect.addEventListener('change', () => this.applyFilters());
        
        // Sort change listeners
        this.sortBy.addEventListener('change', () => this.applyFilters());
        this.sortOrder.addEventListener('change', () => this.applyFilters());
    }
    
    async initializeViewer() {
        this.setStatus('Initializing viewer...', 'loading');
        try {
            await this.loadDataSources();
            this.setStatus('Select a data source and trait/role to begin', '');
        } catch (error) {
            console.error('Error initializing viewer:', error);
            this.setStatus('Error initializing viewer', 'error');
        }
    }
    
    async loadDataSources() {
        const dataSources = [
            { value: 'roles', label: 'Roles (Unique Questions)' },
            { value: 'roles_240', label: 'Roles (Shared Questions)' },
            { value: 'traits', label: 'Traits (Unique Questions)' },
            { value: 'traits_240', label: 'Traits (Shared Questions)' }
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
            this.traitRoleSelect.innerHTML = '<option value="">Select data source first</option>';
            return;
        }
        
        this.currentDataSource = dataSource;
        this.setStatus('Loading trait/role list...', 'loading');
        this.showLoadingBar();
        
        try {
            const traitRoles = await this.discoverTraitRoles(dataSource);
            this.populateTraitRoleSelect(traitRoles);
            this.setStatus(`Found ${traitRoles.length} ${dataSource}`, '');
        } catch (error) {
            console.error('Error loading trait/role list:', error);
            this.setStatus('Error loading trait/role list', 'error');
            this.traitRoleSelect.innerHTML = '<option value="">Error loading list</option>';
        } finally {
            this.hideLoadingBar();
        }
    }
    
    async discoverTraitRoles(dataSource) {
        try {
            const response = await fetch(`data/${dataSource}/index.txt`);
            if (!response.ok) {
                throw new Error(`Failed to load index: ${response.status}`);
            }
            
            const text = await response.text();
            const traitRoles = text.trim().split('\n').filter(line => line.trim());
            
            if (traitRoles.length === 0) {
                throw new Error('No trait/role files found in index.');
            }
            
            return traitRoles.sort();
        } catch (error) {
            console.error('Error loading trait/role index:', error);
            throw new Error(`Failed to load trait/role list: ${error.message}`);
        }
    }
    
    populateTraitRoleSelect(traitRoles) {
        this.traitRoleSelect.innerHTML = '<option value="">Select trait/role...</option>';
        
        traitRoles.forEach(traitRole => {
            const option = document.createElement('option');
            option.value = traitRole;
            option.textContent = traitRole.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            this.traitRoleSelect.appendChild(option);
        });
    }
    
    async loadData() {
        const dataSource = this.dataSourceSelect.value;
        const traitRole = this.traitRoleSelect.value;
        
        if (!dataSource || !traitRole) {
            this.setStatus('Please select both data source and trait/role', 'error');
            return;
        }
        
        this.currentTraitRole = traitRole;
        this.setStatus('Loading data...', 'loading');
        this.showLoadingBar();
        this.loadBtn.disabled = true;
        
        try {
            // Load responses and scores in parallel
            const [responses, scores] = await Promise.all([
                this.loadResponses(dataSource, traitRole),
                this.loadScores(dataSource, traitRole)
            ]);
            
            this.data.responses = responses;
            this.data.scores = scores;
            
            this.updateFiltersForData();
            this.setStatus('Data loaded successfully', 'success');
            this.applyFilters();
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.setStatus(`Error loading data: ${error.message}`, 'error');
            this.clearDisplay();
        } finally {
            this.hideLoadingBar();
            this.loadBtn.disabled = false;
        }
    }
    
    async loadResponses(dataSource, traitRole) {
        const response = await fetch(`data/${dataSource}/responses/${traitRole}.jsonl`);
        if (!response.ok) {
            throw new Error(`Failed to load responses: ${response.status}`);
        }
        
        const text = await response.text();
        const lines = text.trim().split('\n').filter(line => line.trim());
        return lines.map(line => JSON.parse(line));
    }
    
    async loadScores(dataSource, traitRole) {
        // Use extract_labels for roles, extract_scores for traits
        const scoreDir = dataSource.includes('roles') ? 'extract_labels' : 'extract_scores';
        const response = await fetch(`data/${dataSource}/${scoreDir}/${traitRole}.json`);
        if (!response.ok) {
            throw new Error(`Failed to load scores: ${response.status}`);
        }
        
        return await response.json();
    }
    
    updateFiltersForData() {
        // Get unique prompt indices and question indices from loaded data
        const promptIndices = [...new Set(this.data.responses.map(item => item.prompt_index))].sort((a, b) => a - b);
        const questionIndices = [...new Set(this.data.responses.map(item => item.question_index))].sort((a, b) => a - b);
        const availableLabels = [...new Set(this.data.responses.map(item => item.label))];
        
        // Populate prompt index dropdown
        this.promptIndexSelect.innerHTML = '<option value="">All Prompts</option>';
        promptIndices.forEach(index => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `Prompt ${index}`;
            this.promptIndexSelect.appendChild(option);
        });
        
        // Populate question index dropdown
        this.questionIndexSelect.innerHTML = '<option value="">All Questions</option>';
        questionIndices.forEach(index => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `Question ${index}`;
            this.questionIndexSelect.appendChild(option);
        });
        
        // Show/hide negative checkbox based on available labels
        const negCheckboxContainer = this.filterNeg.closest('label');
        if (availableLabels.includes('neg')) {
            negCheckboxContainer.style.display = '';
        } else {
            negCheckboxContainer.style.display = 'none';
            this.filterNeg.checked = false; // Uncheck if hidden
        }
    }
    
    applyFilters() {
        if (!this.data.responses.length) return;
        
        let filtered = this.data.responses.slice();
        
        // Filter by system prompt type
        const showPos = this.filterPos.checked;
        const showNeg = this.filterNeg.checked;
        const showDefault = this.filterDefault.checked;
        
        filtered = filtered.filter(item => {
            if (item.label === 'pos' && !showPos) return false;
            if (item.label === 'neg' && !showNeg) return false;
            if (item.label === 'default' && !showDefault) return false;
            return true;
        });
        
        // Filter by specific prompt index
        const promptIndex = this.promptIndexSelect.value;
        if (promptIndex !== '') {
            filtered = filtered.filter(item => item.prompt_index === parseInt(promptIndex));
        }
        
        // Filter by specific question index
        const questionIndex = this.questionIndexSelect.value;
        if (questionIndex !== '') {
            filtered = filtered.filter(item => item.question_index === parseInt(questionIndex));
        }
        
        // Add scores to filtered data
        filtered = filtered.map(item => ({
            ...item,
            score: this.getScore(item)
        }));
        
        // Sort data
        const sortBy = this.sortBy.value;
        const sortOrder = this.sortOrder.value;
        
        filtered.sort((a, b) => {
            let aVal, bVal;
            
            switch (sortBy) {
                case 'score':
                    aVal = a.score !== null ? a.score : -1;
                    bVal = b.score !== null ? b.score : -1;
                    break;
                case 'prompt_index':
                    aVal = a.prompt_index;
                    bVal = b.prompt_index;
                    break;
                case 'question_index':
                    aVal = a.question_index;
                    bVal = b.question_index;
                    break;
                case 'label':
                    aVal = a.label;
                    bVal = b.label;
                    break;
                default:
                    aVal = a.score !== null ? a.score : -1;
                    bVal = b.score !== null ? b.score : -1;
            }
            
            if (sortOrder === 'desc') {
                return bVal > aVal ? 1 : bVal < aVal ? -1 : 0;
            } else {
                return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
            }
        });
        
        this.filteredData = filtered;
        this.displayData();
        this.updateStats();
    }
    
    getScore(item) {
        const key = `${item.label}_p${item.prompt_index}_q${item.question_index}`;
        return this.data.scores[key] !== undefined ? this.data.scores[key] : null;
    }
    
    getScoreTextColor(score, isRoles = false) {
        if (score === null || score === undefined) {
            return '#000000'; // Black for null scores
        }
        
        if (isRoles) {
            // Role-playing categorical colors (0-3 scale)
            switch (score) {
                case 0: return 'rgb(139, 0, 0)';    // Dark red - REFUSAL
                case 1: return 'rgb(255, 69, 0)';   // Orange-red - NO ROLE-PLAYING 
                case 2: return 'rgb(255, 165, 0)';  // Orange - SOMEWHAT ROLE-PLAYING
                case 3: return 'rgb(34, 139, 34)';  // Forest green - FULLY ROLE-PLAYING
                default: return '#000000';
            }
        }
        
        // Original heat map for traits (0-100 scale)
        const normalizedScore = Math.max(0, Math.min(100, score)) / 100; // Ensure 0-1 range
        
        // Heat map: Red → Orange → Yellow → Blue
        // Score 0: Deep red rgb(204, 0, 0)
        // Score 25: Red-orange rgb(204, 51, 0)
        // Score 50: Orange rgb(204, 102, 0)
        // Score 75: Yellow rgb(204, 204, 0)
        // Score 100: Blue rgb(0, 102, 204)
        
        let red, green, blue;
        
        if (normalizedScore <= 0.25) {
            // 0-25: Red to Red-Orange
            const t = normalizedScore / 0.25;
            red = 204;
            green = Math.round(0 + (51 * t));
            blue = 0;
        } else if (normalizedScore <= 0.5) {
            // 25-50: Red-Orange to Orange
            const t = (normalizedScore - 0.25) / 0.25;
            red = 204;
            green = Math.round(51 + (51 * t));
            blue = 0;
        } else if (normalizedScore <= 0.75) {
            // 50-75: Orange to Yellow
            const t = (normalizedScore - 0.5) / 0.25;
            red = 204;
            green = Math.round(102 + (102 * t));
            blue = 0;
        } else {
            // 75-100: Yellow to Blue
            const t = (normalizedScore - 0.75) / 0.25;
            red = Math.round(204 - (204 * t));
            green = Math.round(204 - (102 * t));
            blue = Math.round(0 + (204 * t));
        }
        
        return `rgb(${red}, ${green}, ${blue})`;
    }
    
    displayData() {
        if (!this.filteredData.length) {
            this.responsesContainer.innerHTML = '<div class="no-data">No data matches current filters</div>';
            return;
        }
        
        let html = '<div class="responses-list">';
        this.filteredData.forEach((item, index) => {
            html += this.renderResponseItem(item, index);
        });
        html += '</div>';
        
        this.responsesContainer.innerHTML = html;
    }
    
    renderResponseItem(item, index) {
        const isRoles = this.currentDataSource && this.currentDataSource.includes('roles');
        let scoreDisplay;
        
        if (item.score !== null && isRoles) {
            // Show categorical labels for roles
            const labels = {
                0: 'REFUSAL',
                1: 'NO ROLE-PLAYING',
                2: 'SOMEWHAT ROLE-PLAYING', 
                3: 'FULLY ROLE-PLAYING'
            };
            scoreDisplay = labels[item.score] || 'Unknown';
        } else {
            scoreDisplay = item.score !== null ? item.score : 'N/A';
        }
        
        const response = item.conversation?.[1]?.content || 'No response available';
        const systemPrompt = item.system_prompt || '';
        const question = item.question || '';
        const scoreTextColor = this.getScoreTextColor(item.score, isRoles);
        
        return `
            <div class="response-item">
                <div class="response-header">
                    <div class="response-score">Score: <span style="color: ${scoreTextColor};">${scoreDisplay}</span></div>
                    <div class="response-meta">
                        ${item.label} | P${item.prompt_index} | Q${item.question_index}
                    </div>
                </div>
                
                <div class="response-system-prompt">
                    <strong>System Prompt:</strong> ${this.processTextFormatting(systemPrompt)}
                </div>
                
                <div class="response-question">
                    <strong>${this.processTextFormatting(question)}</strong>
                </div>
                
                <div class="response-text">
                    ${this.processTextFormatting(response)}
                </div>
            </div>
        `;
    }
    
    
    updateStats() {
        if (!this.filteredData.length) {
            this.stats.style.display = 'none';
            return;
        }
        
        const isRoles = this.currentDataSource && this.currentDataSource.includes('roles');
        const totalItems = this.filteredData.length;
        const scores = this.filteredData.map(item => item.score).filter(s => s !== null);
        
        let avgScore, minScore, maxScore, scoreBreakdown = '';
        
        if (isRoles && scores.length > 0) {
            // For roles, show distribution of categorical scores
            avgScore = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2);
            minScore = Math.min(...scores);
            maxScore = Math.max(...scores);
            
            const roleCounts = { 0: 0, 1: 0, 2: 0, 3: 0 };
            scores.forEach(score => {
                roleCounts[score] = (roleCounts[score] || 0) + 1;
            });
            
            const roleLabels = {
                0: 'Refusal', 
                1: 'No Role-Playing', 
                2: 'Somewhat Role-Playing',
                3: 'Fully Role-Playing'
            };
            
            scoreBreakdown = Object.entries(roleCounts)
                .filter(([score, count]) => count > 0)
                .map(([score, count]) => `${roleLabels[score]}: ${count}`)
                .join(', ');
        } else {
            // For traits, use original numeric statistics
            avgScore = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : 'N/A';
            minScore = scores.length > 0 ? Math.min(...scores) : 'N/A';
            maxScore = scores.length > 0 ? Math.max(...scores) : 'N/A';
        }
        
        const labelCounts = {};
        this.filteredData.forEach(item => {
            labelCounts[item.label] = (labelCounts[item.label] || 0) + 1;
        });
        
        let statsContent = `
            <strong>Statistics:</strong><br>
            Total Items: ${totalItems}<br>
            Average Score: ${avgScore}<br>
            Score Range: ${minScore} - ${maxScore}<br>
        `;
        
        if (scoreBreakdown) {
            statsContent += `Score Distribution: ${scoreBreakdown}<br>`;
        }
        
        statsContent += `Types: ${Object.entries(labelCounts).map(([k, v]) => `${k}: ${v}`).join(', ')}`;
        
        this.stats.innerHTML = statsContent;
        this.stats.style.display = 'block';
    }
    
    clearDisplay() {
        this.responsesContainer.innerHTML = '<div class="no-data">No data loaded</div>';
        this.stats.style.display = 'none';
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
    
    processTextFormatting(text) {
        if (!text) return '';
        
        // First escape HTML to prevent XSS
        const div = document.createElement('div');
        div.textContent = text;
        let processed = div.innerHTML;
        
        // Process markdown formatting
        processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        processed = processed.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, '<em>$1</em>');
        
        // Process bullet points and lists
        processed = this.processLists(processed);
        
        // Convert newlines to <br> tags (do this after list processing)
        processed = processed.replace(/\n/g, '<br>');
        
        return processed;
    }
    
    processLists(text) {
        const lines = text.split('\n');
        const result = [];
        let currentList = null;
        let listType = null;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Check for unordered list items (- or *)
            const unorderedMatch = line.match(/^[-*]\s+(.+)$/);
            // Check for ordered list items (1., 2., etc.)
            const orderedMatch = line.match(/^\d+\.\s+(.+)$/);
            
            if (unorderedMatch) {
                if (listType !== 'ul') {
                    // Close previous list if different type
                    if (currentList) {
                        result.push(`</${listType}>`);
                    }
                    // Start new unordered list
                    result.push('<ul>');
                    listType = 'ul';
                    currentList = [];
                }
                result.push(`<li>${unorderedMatch[1]}</li>`);
            } else if (orderedMatch) {
                if (listType !== 'ol') {
                    // Close previous list if different type
                    if (currentList) {
                        result.push(`</${listType}>`);
                    }
                    // Start new ordered list
                    result.push('<ol>');
                    listType = 'ol';
                    currentList = [];
                }
                result.push(`<li>${orderedMatch[1]}</li>`);
            } else {
                // Not a list item
                if (currentList) {
                    // Close current list
                    result.push(`</${listType}>`);
                    currentList = null;
                    listType = null;
                }
                // Add non-list line
                if (line || i === 0) { // Keep empty lines except at start
                    result.push(line);
                }
            }
        }
        
        // Close any remaining open list
        if (currentList) {
            result.push(`</${listType}>`);
        }
        
        return result.join('\n');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.traitsRolesViewer = new TraitsRolesViewer();
});