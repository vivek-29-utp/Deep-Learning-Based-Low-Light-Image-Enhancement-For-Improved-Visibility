// Zero-DCE++ Web Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const resultsSection = document.querySelector('.results-section');
    const loadingSection = document.querySelector('.loading');
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    uploadBtn.addEventListener('click', function() {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
    
    function handleFileUpload(file) {
        // Validate file type
        if (!file.type.match('image.*')) {
            showMessage('Please upload an image file (JPG, PNG, etc.)', 'error');
            return;
        }
        
        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showMessage('File size must be less than 10MB', 'error');
            return;
        }
        
        // Show loading
        showLoading();
        
        // Create form data
        const formData = new FormData();
        formData.append('image', file);
        
        // Upload image
        fetch('/enhance', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.success) {
                displayResults(data);
            } else {
                showMessage(data.error || 'Enhancement failed', 'error');
            }
        })
        .catch(error => {
            hideLoading();
            showMessage('Upload failed: ' + error.message, 'error');
        });
    }
    
    function showLoading() {
        loadingSection.style.display = 'block';
        resultsSection.style.display = 'none';
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Processing...';
    }
    
    function hideLoading() {
        loadingSection.style.display = 'none';
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Enhance Image';
    }
    
    function displayResults(data) {
        const originalImg = document.getElementById('originalImage');
        const enhancedImg = document.getElementById('enhancedImage');
        const downloadBtn = document.getElementById('downloadBtn');
        
        originalImg.src = data.original_image;
        enhancedImg.src = data.enhanced_image;
        downloadBtn.href = data.enhanced_image;
        
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        showMessage('Image enhanced successfully!', 'success');
    }
    
    function showMessage(message, type) {
        // Remove existing messages
        const existingMessage = document.querySelector('.error-message, .success-message');
        if (existingMessage) {
            existingMessage.remove();
        }
        
        // Create new message
        const messageDiv = document.createElement('div');
        messageDiv.className = type + '-message';
        messageDiv.textContent = message;
        
        // Insert after header
        const header = document.querySelector('.header');
        header.parentNode.insertBefore(messageDiv, header.nextSibling);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            messageDiv.remove();
        }, 5000);
    }
    
    // Image preview on hover
    document.querySelectorAll('.image-container img').forEach(img => {
        img.addEventListener('mouseenter', function() {
            this.style.cursor = 'zoom-in';
        });
        
        img.addEventListener('click', function() {
            // Create modal for full-size view
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.9);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                cursor: zoom-out;
            `;
            
            const modalImg = document.createElement('img');
            modalImg.src = this.src;
            modalImg.style.cssText = `
                max-width: 90%;
                max-height: 90%;
                object-fit: contain;
            `;
            
            modal.appendChild(modalImg);
            document.body.appendChild(modal);
            
            modal.addEventListener('click', function() {
                modal.remove();
            });
        });
    });
});
