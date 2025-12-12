export class FileUploader {
    constructor(onFileSelect, onFileClear) {
        this.dropZone = document.getElementById('drop-zone');
        this.fileInput = document.getElementById('file-input');
        this.fileInfo = document.getElementById('file-info');
        this.filenameDisplay = document.getElementById('filename');
        this.removeFileBtn = document.getElementById('remove-file');

        this.onFileSelect = onFileSelect;
        this.onFileClear = onFileClear;

        this.init();
    }

    init() {
        // Drag events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, (e) => this.preventDefaults(e), false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => this.highlight(), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => this.unhighlight(), false);
        });

        this.dropZone.addEventListener('drop', (e) => this.handleDrop(e), false);
        this.dropZone.addEventListener('click', () => this.handleClick());
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) this.handleFile(e.target.files[0]);
        });

        if (this.removeFileBtn) {
            this.removeFileBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.clearFile();
            });
        }
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight() {
        if (!this.dropZone.classList.contains('disabled')) {
            this.dropZone.classList.add('active');
        }
    }

    unhighlight() {
        this.dropZone.classList.remove('active');
    }

    handleClick() {
        if (!this.dropZone.classList.contains('disabled')) {
            this.fileInput.click();
        }
    }

    handleDrop(e) {
        if (this.dropZone.classList.contains('disabled')) return;
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) this.handleFile(files[0]);
    }

    handleFile(file) {
        const validExtensions = ['.pdf', '.doc', '.docx', '.txt'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(fileExtension)) {
            alert('Error: Unsupported file format! Please upload only .pdf, .doc, .docx or .txt files.');
            this.clearFile();
            return;
        }

        this.filenameDisplay.textContent = file.name;
        this.fileInfo.style.display = 'inline-flex';
        if (this.onFileSelect) this.onFileSelect(file);
    }

    clearFile() {
        this.fileInput.value = '';
        this.fileInfo.style.display = 'none';
        this.filenameDisplay.textContent = '';
        if (this.onFileClear) this.onFileClear();
    }

    setDisabled(disabled) {
        if (disabled) {
            this.dropZone.classList.add('disabled');
            this.dropZone.style.opacity = '0.5';
            this.dropZone.style.cursor = 'not-allowed';
        } else {
            this.dropZone.classList.remove('disabled');
            this.dropZone.style.opacity = '1';
            this.dropZone.style.cursor = 'pointer';
        }
    }
}
