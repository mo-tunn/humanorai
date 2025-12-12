import { ApiService } from './modules/api.js';
import { UIManager } from './modules/ui.js';
import { FileUploader } from './modules/dragDrop.js';

document.addEventListener('DOMContentLoaded', () => {
    const apiService = new ApiService();
    const uiManager = new UIManager();
    const articleInput = document.getElementById('article-input');

    let selectedFile = null;

    // File Upload Handler
    const fileUploader = new FileUploader(
        (file) => {
            // On file select
            selectedFile = file;
            articleInput.value = '';
            articleInput.disabled = true;
            articleInput.placeholder = "📁 File selected. Remove the file above to enter text.";
            articleInput.style.opacity = "0.6";
            articleInput.style.cursor = "not-allowed";
        },
        () => {
            // On file clear
            selectedFile = null;
            articleInput.disabled = false;
            articleInput.placeholder = "Paste the article you want to analyze here...";
            articleInput.style.opacity = "1";
            articleInput.style.cursor = "text";
        }
    );

    // Text Input Handler
    articleInput.addEventListener('input', () => {
        const hasText = articleInput.value.trim().length > 0;
        fileUploader.setDisabled(hasText);
    });

    // Analyze Button Handler
    const analyzeButton = document.getElementById('analyze-button');
    analyzeButton.addEventListener('click', async () => {
        console.log("Analyze button clicked");
        const articleText = articleInput.value.trim();
        // Check if file is selected (rudimentary check based on UI state)
        const hasFile = document.getElementById('file-info').style.display === 'inline-flex';
        console.log("hasFile:", hasFile, "selectedFile:", selectedFile);

        if (hasFile) {
            const fileName = document.getElementById('filename').textContent;
            const validExtensions = ['.pdf', '.doc', '.docx', '.txt'];
            const fileExtension = '.' + fileName.split('.').pop().toLowerCase();

            if (!validExtensions.includes(fileExtension)) {
                alert('Error: Unsupported file format! Please upload only .pdf, .doc, .docx or .txt files.');
                return;
            }
        }

        if (articleText.length === 0 && !hasFile) {
            articleInput.focus();
            return;
        }

        if (articleText.length > 0 && articleText.length < 50) {
            alert("Minimum 50 characters required for analysis.");
            return;
        }

        uiManager.setLoading(true);

        try {
            let data;
            if (hasFile && selectedFile) {
                console.log("Sending file to backend...");
                data = await apiService.predictFile(selectedFile);
            } else {
                console.log("Sending text to backend...");
                data = await apiService.predict(articleText);
            }
            uiManager.displayResults(data);

        } catch (error) {
            console.error(error);
            alert("An error occurred during analysis: " + error.message);
        } finally {
            uiManager.setLoading(false);
        }
    });

    // Modal Handlers
    const modal = document.getElementById('accuracy-modal');
    const openBtn = document.getElementById('open-accuracy-modal');
    const closeBtn = document.getElementById('close-modal');

    // Close modal when clicking outside content
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    if (openBtn) {
        openBtn.addEventListener('click', (e) => {
            e.preventDefault();
            modal.style.display = 'flex';
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }
});
