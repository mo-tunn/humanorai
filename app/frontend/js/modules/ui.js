export class UIManager {
    constructor() {
        this.resultsCard = document.getElementById('results-card');
        this.resultsContent = document.getElementById('results-content');
        this.analyzeButton = document.getElementById('analyze-button');
    }

    setLoading(isLoading) {
        if (isLoading) {
            this.analyzeButton.disabled = true;
            this.analyzeButton.textContent = "Analyzing...";
            this.resultsCard.style.display = 'none';
        } else {
            this.analyzeButton.disabled = false;
            this.analyzeButton.textContent = "Start Analysis";
        }
    }

    displayResults(data) {
        const html = `
            <table class="glass-table">
                <thead>
                    <tr>
                        <th style="width: 35%;">Model <span style="font-size: 0.8em; font-weight: normal; color: var(--text-secondary);">(Accuracy)</span></th>
                        <th>AI Prob.</th>
                        <th>Human Prob.</th>
                        <th style="text-align: right;">Decision</th>
                    </tr>
                </thead>
                <tbody>
                    ${this._generateTableRows(data.individual_results)}
                </tbody>
            </table>
            ${this._generateFinalVerdict(data)}
        `;

        this.resultsContent.innerHTML = html;
        this.resultsCard.style.display = 'block';
        this.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    _generateTableRows(results) {
        const accuracies = {
            "Logistic Regression": "98.59%",
            "Random Forest": "97.59%",
            "Linear SVM": "98.83%",
            "Neural Network": "98.71%",
            "AdaBoost": "83.61%",
            "Gradient Boosting": "89.95%",
            "Decision Tree": "89.20%",
            "Naive Bayes": "95.56%"
        };

        return results.map(result => {
            const { model, probability_percent, decision } = result;
            const { statusClass, icon } = this._getStatusStyles(probability_percent, decision);

            const aiProb = probability_percent;
            const humanProb = 100 - aiProb;
            const accuracy = accuracies[model] || "~94%";

            return `
                <tr>
                    <td>
                        <div style="font-weight: 600;">${model}</div>
                        <div style="font-size: 0.8em; color: var(--primary-color); background: rgba(37, 99, 235, 0.1); display: inline-block; padding: 2px 6px; border-radius: 4px; margin-top: 2px;">
                            ${accuracy} Acc.
                        </div>
                    </td>
                    <td style="color: #ef4444; font-weight: 600;">%${aiProb.toFixed(2)}</td>
                    <td style="color: #10b981; font-weight: 600;">%${humanProb.toFixed(2)}</td>
                    <td>
                        <span class="status-badge ${statusClass}">
                            ${icon} ${decision}
                        </span>
                    </td>
                </tr>
            `;
        }).join('');
    }

    _generateFinalVerdict(data) {
        const avgPercent = data.ensemble_average_percent;
        const decision = data.ensemble_decision;


        // Custom styling for the final verdict
        let verdictClass = "verdict-uncertain";
        let verdictColor = "var(--text-secondary)";

        // Default Icon (Uncertain - Pulse)
        let icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="status-icon"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg>`;

        if (avgPercent > 60) {
            verdictClass = "verdict-ai";
            verdictColor = "#ef4444";
            // AI Icon (Custom Tech Brain)
            icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 48 48" class="status-icon"><path d="M45.6,18.7,41,14.9V7.5a1,1,0,0,0-.6-.9L30.5,2.1h-.4l-.6.2L24,5.9,18.5,2.2,17.9,2h-.4L7.6,6.6a1,1,0,0,0-.6.9v7.4L2.4,18.7a.8.8,0,0,0-.4.8v9H2a.8.8,0,0,0,.4.8L7,33.1v7.4a1,1,0,0,0,.6.9l9.9,4.5h.4l.6-.2L24,42.1l5.5,3.7.6.2h.4l9.9-4.5a1,1,0,0,0,.6-.9V33.1l4.6-3.8a.8.8,0,0,0,.4-.7V19.4h0A.8.8,0,0,0,45.6,18.7Zm-5.1,6.8H42v1.6l-3.5,2.8-.4.3-.4-.2a1.4,1.4,0,0,0-2,.7,1.5,1.5,0,0,0,.6,2l.7.3h0v5.4l-6.6,3.1-4.2-2.8-.7-.5V25.5H27a1.5,1.5,0,0,0,0-3H25.5V9.7l.7-.5,4.2-2.8L37,9.5v5.4h0l-.7.3a1.5,1.5,0,0,0-.6,2,1.4,1.4,0,0,0,1.3.9l.7-.2.4-.2.4.3L42,20.9v1.6H40.5a1.5,1.5,0,0,0,0,3ZM21,25.5h1.5V38.3l-.7.5-4.2,2.8L11,38.5V33.1h0l.7-.3a1.5,1.5,0,0,0,.6-2,1.4,1.4,0,0,0-2-.7l-.4.2-.4-.3L6,27.1V25.5H7.5a1.5,1.5,0,0,0,0-3H6V20.9l3.5-2.8.4-.3.4.2.7.2a1.4,1.4,0,0,0,1.3-.9,1.5,1.5,0,0,0-.6-2L11,15h0V9.5l6.6-3.1,4.2,2.8.7.5V22.5H21a1.5,1.5,0,0,0,0,3Z"/><path d="M13.9,9.9a1.8,1.8,0,0,0,0,2.2l2.6,2.5v2.8l-4,4v5.2l4,4v2.8l-2.6,2.5a1.8,1.8,0,0,0,0,2.2,1.5,1.5,0,0,0,1.1.4,1.5,1.5,0,0,0,1.1-.4l3.4-3.5V29.4l-4-4V22.6l4-4V13.4L16.1,9.9A1.8,1.8,0,0,0,13.9,9.9Z"/><path d="M31.5,14.6l2.6-2.5a1.8,1.8,0,0,0,0-2.2,1.8,1.8,0,0,0-2.2,0l-3.4,3.5v5.2l4,4v2.8l-4,4v5.2l3.4,3.5a1.7,1.7,0,0,0,2.2,0,1.8,1.8,0,0,0,0-2.2l-2.6-2.5V30.6l4-4V21.4l-4-4Z"/></svg>`;
        } else if (avgPercent < 40) {
            verdictClass = "verdict-human";
            verdictColor = "#10b981";
            // Human Icon (User)
            icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="status-icon"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" /></svg>`;
        }

        return `
            <div class="final-verdict-card ${verdictClass}">
                <div class="verdict-header">ENSEMBLE RESULT (FINAL)</div>
                <div class="verdict-body">
                    <div class="verdict-score">
                        <span class="percentage">%${avgPercent.toFixed(2)}</span>
                        <span class="label">AI Probability</span>
                    </div>
                    <div class="verdict-decision" style="color: ${verdictColor}">
                        <span class="icon-wrapper">${icon}</span>
                        <span class="text">${decision}</span>
                    </div>
                </div>
            </div>
        `;
    }



    _getStatusStyles(percent, decisionText) {
        let statusClass = "status-uncertain";
        // Default Icon (Uncertain - Pulse)
        let icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="status-icon-sm"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg>`;

        // Logic based on percentage or decision text
        if (percent > 60) {
            statusClass = "status-ai";
            // AI Icon (Custom Tech Brain)
            icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 48 48" class="status-icon-sm"><path d="M45.6,18.7,41,14.9V7.5a1,1,0,0,0-.6-.9L30.5,2.1h-.4l-.6.2L24,5.9,18.5,2.2,17.9,2h-.4L7.6,6.6a1,1,0,0,0-.6.9v7.4L2.4,18.7a.8.8,0,0,0-.4.8v9H2a.8.8,0,0,0,.4.8L7,33.1v7.4a1,1,0,0,0,.6.9l9.9,4.5h.4l.6-.2L24,42.1l5.5,3.7.6.2h.4l9.9-4.5a1,1,0,0,0,.6-.9V33.1l4.6-3.8a.8.8,0,0,0,.4-.7V19.4h0A.8.8,0,0,0,45.6,18.7Zm-5.1,6.8H42v1.6l-3.5,2.8-.4.3-.4-.2a1.4,1.4,0,0,0-2,.7,1.5,1.5,0,0,0,.6,2l.7.3h0v5.4l-6.6,3.1-4.2-2.8-.7-.5V25.5H27a1.5,1.5,0,0,0,0-3H25.5V9.7l.7-.5,4.2-2.8L37,9.5v5.4h0l-.7.3a1.5,1.5,0,0,0-.6,2,1.4,1.4,0,0,0,1.3.9l.7-.2.4-.2.4.3L42,20.9v1.6H40.5a1.5,1.5,0,0,0,0,3ZM21,25.5h1.5V38.3l-.7.5-4.2,2.8L11,38.5V33.1h0l.7-.3a1.5,1.5,0,0,0,.6-2,1.4,1.4,0,0,0-2-.7l-.4.2-.4-.3L6,27.1V25.5H7.5a1.5,1.5,0,0,0,0-3H6V20.9l3.5-2.8.4-.3.4.2.7.2a1.4,1.4,0,0,0,1.3-.9,1.5,1.5,0,0,0-.6-2L11,15h0V9.5l6.6-3.1,4.2,2.8.7.5V22.5H21a1.5,1.5,0,0,0,0,3Z"/><path d="M13.9,9.9a1.8,1.8,0,0,0,0,2.2l2.6,2.5v2.8l-4,4v5.2l4,4v2.8l-2.6,2.5a1.8,1.8,0,0,0,0,2.2,1.5,1.5,0,0,0,1.1.4,1.5,1.5,0,0,0,1.1-.4l3.4-3.5V29.4l-4-4V22.6l4-4V13.4L16.1,9.9A1.8,1.8,0,0,0,13.9,9.9Z"/><path d="M31.5,14.6l2.6-2.5a1.8,1.8,0,0,0,0-2.2,1.8,1.8,0,0,0-2.2,0l-3.4,3.5v5.2l4,4v2.8l-4,4v5.2l3.4,3.5a1.7,1.7,0,0,0,2.2,0,1.8,1.8,0,0,0,0-2.2l-2.6-2.5V30.6l4-4V21.4l-4-4Z"/></svg>`;
        } else if (percent < 40) {
            statusClass = "status-human";
            // Human Icon (User)
            icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="status-icon-sm"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" /></svg>`;
        }

        return { statusClass, icon };
    }
}
