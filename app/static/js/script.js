document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const codeForm = document.getElementById('codeForm');
    const resultsSection = document.getElementById('results');
    const loadingIndicator = document.getElementById('loading');
    const resultsContent = document.getElementById('results-content');
    const examplesContainer = document.getElementById('examples-container');
    
    // Load examples when the page loads
    loadExamples();
    
    // Form submission
    codeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        analyzeCode();
    });
    
    // Function to load example code snippets
    function loadExamples() {
        fetch('/api/examples/all')
            .then(response => response.json())
            .then(examples => {
                examplesContainer.innerHTML = '';
                
                examples.forEach((example, index) => {
                    const exampleCard = document.createElement('div');
                    exampleCard.className = 'example-card';
                    exampleCard.innerHTML = `
                        <h4>Example ${index + 1}: ${example.description}</h4>
                        <p>Language: ${example.language}</p>
                    `;
                    
                    exampleCard.addEventListener('click', function() {
                        document.getElementById('language').value = example.language;
                        document.getElementById('code').value = example.code;
                        document.getElementById('error').value = example.error_message;
                        
                        // Scroll to the top of the form
                        codeForm.scrollIntoView({ behavior: 'smooth' });
                    });
                    
                    examplesContainer.appendChild(exampleCard);
                });
            })
            .catch(error => {
                console.error('Error loading examples:', error);
                examplesContainer.innerHTML = '<p>Failed to load examples. Please try again later.</p>';
            });
    }
    
    // Function to analyze code
    function analyzeCode() {
        const language = document.getElementById('language').value;
        const code = document.getElementById('code').value;
        const errorMessage = document.getElementById('error').value;
        
        // Validate inputs
        if (!code.trim()) {
            alert('Please enter your code.');
            return;
        }
        
        if (!errorMessage.trim()) {
            alert('Please enter the error message.');
            return;
        }
        
        // Show loading indicator and results section
        resultsSection.style.display = 'block';
        loadingIndicator.style.display = 'block';
        resultsContent.style.display = 'none';
        
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Send data to the server
        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                language: language,
                code: code,
                error_message: errorMessage
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('error-type').textContent = 'Analysis Error';
            document.getElementById('root-cause').textContent = 'Failed to analyze the code.';
            document.getElementById('explanation').textContent = 'There was an error processing your request. Please try again or check your inputs.';
            document.getElementById('solutions').innerHTML = '<p>Error details: ' + error.message + '</p>';
            
            // Hide loading indicator and show results
            loadingIndicator.style.display = 'none';
            resultsContent.style.display = 'block';
        });
    }
    
    // Function to display analysis results
    function displayResults(data) {
        // Set the content
        document.getElementById('error-type').textContent = data.analysis?.error_type || 'Unknown Error';
        document.getElementById('root-cause').textContent = data.analysis?.root_cause || 'Could not determine root cause';
        document.getElementById('explanation').textContent = data.analysis?.error_type ? 
            `Error occurred on line ${data.analysis.line_number || 'unknown'}. ${data.analysis.root_cause || ''}` : 
            'No explanation available';
        
        // Display solutions
        const solutionsContainer = document.getElementById('solutions');
        solutionsContainer.innerHTML = '';
        
        if (data.solutions && data.solutions.length > 0) {
            data.solutions.forEach((solution, index) => {
                const solutionElement = document.createElement('div');
                solutionElement.className = 'solution';
                
                // Create solution header
                const solutionHeader = document.createElement('h4');
                solutionHeader.textContent = `Solution ${index + 1}`;
                solutionElement.appendChild(solutionHeader);
                
                // Create solution description
                const solutionDesc = document.createElement('p');
                solutionDesc.textContent = solution.description || 'No description available';
                solutionElement.appendChild(solutionDesc);
                
                // Create code snippet if available
                if (solution.code) {
                    const codeBlock = document.createElement('pre');
                    const codeElement = document.createElement('code');
                    codeElement.className = document.getElementById('language').value;
                    codeElement.textContent = solution.code;
                    codeBlock.appendChild(codeElement);
                    solutionElement.appendChild(codeBlock);
                    
                    // Initialize syntax highlighting
                    hljs.highlightElement(codeElement);
                }
                
                solutionsContainer.appendChild(solutionElement);
            });
        } else {
            solutionsContainer.innerHTML = '<p>No solutions available</p>';
        }
        
        // Hide loading indicator and show results
        loadingIndicator.style.display = 'none';
        resultsContent.style.display = 'block';
    }
});